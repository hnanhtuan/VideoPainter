import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
GRADIO_TEMP_DIR = "./tmp_gradio"
os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR
import warnings
warnings.filterwarnings("ignore")
import argparse
from typing import Literal
import json
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogvideoXBranchModel,
    CogVideoXTransformer3DModel,
    CogVideoXI2VDualInpaintPipeline,
    CogVideoXI2VDualInpaintAnyLPipeline,
    FluxFillPipeline
)
import cv2
from openai import OpenAI
from diffusers.utils import export_to_video, load_image, load_video
from PIL import Image
from safetensors import safe_open
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

def load_video_model(
    model_path,
    inpainting_branch,
    id_adapter,
    device="cuda:0",
    dtype=torch.bfloat16,
    cpu_offload=False,
    offload_mode=None,
):
    offload_mode = "model" if cpu_offload and offload_mode is None else offload_mode
    if offload_mode not in {None, "model", "sequential"}:
        raise ValueError('offload_mode must be None, "model", or "sequential".')
    if offload_mode and torch.device(device).type != "cuda":
        raise ValueError("CPU offloading requires a CUDA execution device.")

    # In offload mode, construct float16 modules on CPU. Accelerate hooks move
    # them to CUDA only for execution, avoiding an eager full-model VRAM spike.
    branch = CogvideoXBranchModel.from_pretrained(
        inpainting_branch, torch_dtype=dtype
    )
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype,
        id_pool_resample_learnable=True,
    )
    if offload_mode is None:
        branch.to(device, dtype=dtype)
        transformer.to(device, dtype=dtype)

    pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
        model_path,
        branch=branch,
        transformer=transformer,
        torch_dtype=dtype,
    )
    pipe.load_lora_weights(
        id_adapter,
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="test_1",
        target_modules=["transformer"],
    )
    print(f"list_adapters_component_wise: {pipe.get_list_adapters()}")

    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.branch.requires_grad_(False)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    if offload_mode == "sequential":
        print("Enabling sequential CPU offloading for VideoPainter pipeline...")
        pipe.enable_sequential_cpu_offload(device=device)
    elif offload_mode == "model":
        print("Enabling model CPU offloading for VideoPainter pipeline...")
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def load_flux_model(
    img_inpainting_model,
    device="cuda:0",
    dtype=torch.bfloat16,
    cpu_offload=False,
    offload_mode=None,
):
    offload_mode = "model" if cpu_offload and offload_mode is None else offload_mode
    if offload_mode not in {None, "model", "sequential"}:
        raise ValueError('offload_mode must be None, "model", or "sequential".')
    if offload_mode and torch.device(device).type != "cuda":
        raise ValueError("CPU offloading requires a CUDA execution device.")
    pipe = FluxFillPipeline.from_pretrained(
        img_inpainting_model, torch_dtype=dtype
    )
    if offload_mode == "sequential":
        print("Enabling sequential CPU offloading for FLUX inpainting pipeline...")
        pipe.enable_sequential_cpu_offload(device=device)
    elif offload_mode == "model":
        print("Enabling model CPU offloading for FLUX inpainting pipeline...")
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    return pipe


def load_model(
    model_path,
    inpainting_branch,
    img_inpainting_model,
    id_adapter,
    device="cuda:0",
    dtype=torch.bfloat16,
    cpu_offload=False,
    video_device=None,
    flux_device=None,
    video_dtype=None,
    flux_dtype=None,
    video_offload_mode=None,
    flux_offload_mode=None,
):
    """Load both pipelines for the Gradio app and legacy callers."""
    video_device = device if video_device is None else video_device
    flux_device = device if flux_device is None else flux_device
    video_dtype = dtype if video_dtype is None else video_dtype
    flux_dtype = dtype if flux_dtype is None else flux_dtype

    video_pipe = load_video_model(
        model_path=model_path,
        inpainting_branch=inpainting_branch,
        id_adapter=id_adapter,
        device=video_device,
        dtype=video_dtype,
        cpu_offload=cpu_offload,
        offload_mode=video_offload_mode,
    )
    flux_pipe = load_flux_model(
        img_inpainting_model=img_inpainting_model,
        device=flux_device,
        dtype=flux_dtype,
        cpu_offload=cpu_offload,
        offload_mode=flux_offload_mode,
    )
    return video_pipe, flux_pipe



def _pipeline_has_offload_hooks(pipe):
    """Return True when Accelerate CPU-offload hooks are attached to components."""
    for component in getattr(pipe, "components", {}).values():
        if not isinstance(component, torch.nn.Module):
            continue
        if hasattr(component, "_hf_hook"):
            return True
        if any(hasattr(module, "_hf_hook") for module in component.modules()):
            return True
    return bool(getattr(pipe, "_all_hooks", []))

def generate_flux_frame(
    images,
    masks,
    pipe_img_inpainting,
    image_inpainting_prompt,
    seed=42,
    dilate_size=16,
    inference_device="cuda",
):
    """Run FLUX on the first frame and return it with prepared video masks."""
    prepared_images = [image.copy() for image in images]
    prepared_masks = [mask.copy() for mask in masks]
    if not prepared_images or len(prepared_images) != len(prepared_masks):
        raise ValueError("Images and masks must be non-empty and have equal lengths.")

    os.makedirs(f"{GRADIO_TEMP_DIR}/inpaint", exist_ok=True)
    prepared_images[0].save(f"{GRADIO_TEMP_DIR}/inpaint/first_frame.png")
    prepared_masks[0].save(f"{GRADIO_TEMP_DIR}/inpaint/first_mask.png")
    prepared_masks[-1].save(f"{GRADIO_TEMP_DIR}/inpaint/last_mask.png")

    print(f"Dilating the mask with size {dilate_size}...")
    kernel = np.ones((dilate_size, dilate_size), dtype=np.uint8)
    for index, mask in enumerate(prepared_masks):
        dilated = cv2.dilate(np.asarray(mask, dtype=np.uint8), kernel)
        prepared_masks[index] = Image.fromarray(dilated).convert("RGB")

    prepared_masks[0].save(f"{GRADIO_TEMP_DIR}/inpaint/first_mask_dilate.png")
    prepared_masks[-1].save(f"{GRADIO_TEMP_DIR}/inpaint/last_mask_dilate.png")
    print(f"Image inpainting prompt: {image_inpainting_prompt}")

    is_offloaded = _pipeline_has_offload_hooks(pipe_img_inpainting)
    if not is_offloaded:
        pipe_img_inpainting.to(inference_device)
    image_inpainting = pipe_img_inpainting(
        prompt=image_inpainting_prompt,
        image=prepared_images[0],
        mask_image=prepared_masks[0],
        height=prepared_images[0].size[1],
        width=prepared_images[0].size[0],
        guidance_scale=30,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]
    if not is_offloaded and torch.device(inference_device).type == "cuda":
        pipe_img_inpainting.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    image_inpainting.save(f"{GRADIO_TEMP_DIR}/inpaint/first_frame_inpainted.png")
    prepared_masks[0] = Image.fromarray(
        np.zeros_like(np.asarray(prepared_masks[0]), dtype=np.uint8)
    ).convert("RGB")
    print(f"Image inpainting done! {np.asarray(image_inpainting).shape}")
    return image_inpainting, prepared_masks


def _frames_to_uint8(frames):
    frames = np.asarray(frames)
    if np.issubdtype(frames.dtype, np.floating):
        return np.clip(frames * 255.0, 0, 255).astype(np.uint8)
    return np.clip(frames, 0, 255).astype(np.uint8)


def _save_generated_frame_images(frames, frame_output_dir):
    if frame_output_dir is None:
        return
    os.makedirs(frame_output_dir, exist_ok=True)
    for index, frame in enumerate(_frames_to_uint8(frames)):
        Image.fromarray(frame).save(os.path.join(frame_output_dir, f"frame_{index:04d}.png"))
    print(f"Saved {len(frames)} VideoPainter frames to: {frame_output_dir}")


def generate_video_frames(
    images,
    masks,
    pipe,
    prompt,
    seed=42,
    cfg_scale=6.0,
    frame_output_dir=None,
):
    """Run VideoPainter using a FLUX-processed first frame.

    When frame_output_dir is provided, decoded RGB frames are written as PNGs
    immediately after the pipeline returns them, before video export happens.
    """
    inpaint_outputs = pipe(
        prompt=prompt,
        image=images[0],
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        use_dynamic_cfg=True,
        guidance_scale=cfg_scale,
        generator=torch.Generator().manual_seed(seed),
        video=images,
        masks=masks,
        strength=1.0,
        replace_gt=True,
        mask_add=True,
        stride=49,
        prev_clip_weight=0.0,
        id_pool_resample_learnable=False,
        output_type="np",
    ).frames[0][1:]
    _save_generated_frame_images(inpaint_outputs, frame_output_dir)
    print(
        f"Video inpainting done! {np.asarray(inpaint_outputs).shape}, "
        f"{np.asarray(inpaint_outputs).min()}, {np.asarray(inpaint_outputs).max()}"
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return inpaint_outputs


def generate_frames(
    images,
    masks,
    pipe,
    pipe_img_inpainting,
    prompt,
    image_inpainting_prompt,
    seed=42,
    cfg_scale=6.0,
    dilate_size=16,
    inference_device="cuda",
):
    """Run the original combined FLUX then VideoPainter workflow."""
    image_inpainting, prepared_masks = generate_flux_frame(
        images=images,
        masks=masks,
        pipe_img_inpainting=pipe_img_inpainting,
        image_inpainting_prompt=image_inpainting_prompt,
        seed=seed,
        dilate_size=dilate_size,
        inference_device=inference_device,
    )
    prepared_images = [image.copy() for image in images]
    prepared_images[0] = image_inpainting
    return generate_video_frames(
        images=prepared_images,
        masks=prepared_masks,
        pipe=pipe,
        prompt=prompt,
        seed=seed,
        cfg_scale=cfg_scale,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--inpainting_branch", type=str, default="")
    parser.add_argument("--img_inpainting_model", type=str, default="../")
    args = parser.parse_args()


    validation_pipeline = load_model(
        model_path=args.model_path,
        inpainting_branch=args.inpainting_branch,
        img_inpainting_model=args.img_inpainting_model
    )
