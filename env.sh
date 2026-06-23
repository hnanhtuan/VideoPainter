# Create a virtual environment using python 3.10 (managed by uv)
uv venv --python 3.10
source .venv/bin/activate

# Install dependencies from requirements.txt
uv pip install -r requirements.txt

# Install editable packages
uv pip install -e ./app --no-build-isolation
uv pip install -e ./diffusers

