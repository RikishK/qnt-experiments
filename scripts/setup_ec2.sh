#!/usr/bin/env bash
# Setup script for g6.48xlarge EC2 instance (8x NVIDIA L4)
# Assumes conda is available and NVIDIA drivers are pre-installed.
# Run as: bash scripts/setup_ec2.sh
set -euo pipefail

ENV_NAME="qnt"

echo "=== Qwen3-32B Boundary V Quantization Experiment - EC2 Setup ==="

# --- Check NVIDIA drivers ---
echo "[1/4] Checking NVIDIA drivers..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# --- Create conda environment ---
echo "[2/4] Creating conda environment '${ENV_NAME}'..."
conda create -n "$ENV_NAME" python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# --- Install PyTorch with CUDA + project deps ---
echo "[3/4] Installing PyTorch + project dependencies..."
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Enable fast HuggingFace downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# --- Verify GPU setup ---
echo "[4/4] Verifying GPU setup..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_mem / 1024**3:.1f} GB)')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo "Next: bash scripts/download_model.sh"
