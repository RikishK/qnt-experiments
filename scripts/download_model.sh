#!/usr/bin/env bash
# Download Qwen3-32B and evaluation datasets
# Run as: bash scripts/download_model.sh
set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen3-32B}"
MODEL_DIR="./models"
DATA_DIR="./data"

echo "=== Downloading model and datasets ==="

# Enable fast HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# --- Download model ---
echo "[1/3] Downloading ${MODEL_ID}..."
mkdir -p "$MODEL_DIR"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '${MODEL_ID}',
    local_dir='${MODEL_DIR}/$(basename $MODEL_ID)',
    ignore_patterns=['*.gguf', '*.bin'],  # prefer safetensors
)
print('Model download complete.')
"

# --- Download evaluation datasets ---
echo "[2/3] Downloading HumanEval and MBPP datasets..."
mkdir -p "$DATA_DIR"
python3 -c "
from datasets import load_dataset

# MBPP sanitized split
print('Downloading MBPP...')
ds = load_dataset('google-research-datasets/mbpp', 'sanitized', split='test')
ds.save_to_disk('${DATA_DIR}/mbpp')
print(f'MBPP: {len(ds)} problems')

# HumanEval
print('Downloading HumanEval...')
ds = load_dataset('openai/openai_humaneval', split='test')
ds.save_to_disk('${DATA_DIR}/humaneval')
print(f'HumanEval: {len(ds)} problems')
"

# --- Download calibration data (wikitext-2 for perplexity + quant calibration) ---
echo "[3/3] Downloading wikitext-2 calibration data..."
python3 -c "
from datasets import load_dataset
print('Downloading wikitext-2...')
ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
ds.save_to_disk('${DATA_DIR}/wikitext2')
print(f'Wikitext-2: {len(ds)} samples')
"

echo ""
echo "=== Downloads complete ==="
echo "Model: ${MODEL_DIR}/$(basename $MODEL_ID)"
echo "Data:  ${DATA_DIR}/"
ls -lh "$DATA_DIR"/
