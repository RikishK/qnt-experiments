#!/usr/bin/env bash
# Setup script for g6.48xlarge EC2 instance (8x NVIDIA L4, Ubuntu)
# Run as: bash scripts/setup_ec2.sh
set -euo pipefail

echo "=== Qwen3-32B Boundary V Quantization Experiment - EC2 Setup ==="

# --- System packages ---
echo "[1/6] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq git curl wget htop nvtop tmux python3-pip python3-venv

# --- NVIDIA drivers (if not already installed on Deep Learning AMI) ---
echo "[2/6] Checking NVIDIA drivers..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "NVIDIA drivers not found. If using Deep Learning AMI, they should be pre-installed."
    echo "For a fresh Ubuntu instance, install drivers manually:"
    echo "  sudo apt-get install -y nvidia-driver-550 nvidia-utils-550"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# --- Python environment ---
echo "[3/6] Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# --- Install PyTorch with CUDA ---
echo "[4/6] Installing PyTorch with CUDA support..."
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# --- Install project dependencies ---
echo "[5/6] Installing project dependencies..."
pip install -r requirements.txt

# Enable fast HuggingFace downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# --- Verify GPU setup ---
echo "[6/6] Verifying GPU setup..."
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
echo "Activate the environment with: source venv/bin/activate"
echo "Next: bash scripts/download_model.sh"
