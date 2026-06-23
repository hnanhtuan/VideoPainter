#!/bin/bash
# Locate the virtual environment relative to this script and activate it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/../.venv" ]; then
    source "$SCRIPT_DIR/../.venv/bin/activate"
elif [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# Load environment variables from .env file if it exists
if [ -f "$SCRIPT_DIR/../.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/../.env" | xargs)
elif [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

CUDA_VISIBLE_DEVICES=0 python app.py \
    --model_path ../ckpt/CogVideoX-5b-I2V \
    --inpainting_branch ../ckpt/VideoPainter/VideoPainter/checkpoints/branch \
    --id_adapter ../ckpt/VideoPainter/VideoPainterID/checkpoints \
    --img_inpainting_model ../ckpt/flux_inp
