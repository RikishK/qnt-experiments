#!/usr/bin/env bash
# Master experiment runner.
# Run as: bash scripts/run_all.sh [--tp 8] [--skip-baseline] [--configs "boundary-v1 boundary-v2"]
set -euo pipefail

# --- Defaults ---
TP_SIZE=8
MODEL_DIR="./models/Qwen3-32B"
DATA_DIR="./data"
RESULTS_DIR="./results"
OUTPUT_DIR="./models/quantized"
SKIP_BASELINE=false
CONFIGS="uniform-int8 uniform-int4 boundary-v1 boundary-v2 boundary-v3 boundary-v4"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --tp) TP_SIZE="$2"; shift 2 ;;
        --skip-baseline) SKIP_BASELINE=true; shift ;;
        --configs) CONFIGS="$2"; shift 2 ;;
        --model) MODEL_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "========================================="
echo "  Boundary V Quantization Experiment"
echo "========================================="
echo "Model:    $MODEL_DIR"
echo "TP size:  $TP_SIZE"
echo "Configs:  $CONFIGS"
echo "========================================="
echo ""

# Activate venv if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# --- Phase 1: Baseline ---
if [ "$SKIP_BASELINE" = false ]; then
    echo "[Phase 1] Running FP16 baseline evaluation..."
    python3 src/evaluate.py \
        --model "$MODEL_DIR" \
        --config-name baseline \
        --data-dir "$DATA_DIR" \
        --output-dir "$RESULTS_DIR" \
        --tp "$TP_SIZE"
    echo "Baseline complete."
    echo ""
fi

# --- Phase 2: Quantize all configs ---
echo "[Phase 2] Quantizing models..."
for CONFIG in $CONFIGS; do
    QUANT_DIR="${OUTPUT_DIR}/${CONFIG}"
    if [ -d "$QUANT_DIR" ] && [ -f "$QUANT_DIR/quant_config.json" ]; then
        echo "  Skipping $CONFIG (already quantized)"
        continue
    fi
    echo "  Quantizing: $CONFIG..."
    python3 src/quantize.py \
        --model "$MODEL_DIR" \
        --config "$CONFIG" \
        --output "$QUANT_DIR" \
        --verbose
    echo "  Done: $CONFIG"
done
echo "All models quantized."
echo ""

# --- Phase 3: Evaluate all configs ---
echo "[Phase 3] Evaluating quantized models..."
for CONFIG in $CONFIGS; do
    QUANT_DIR="${OUTPUT_DIR}/${CONFIG}"
    echo "  Evaluating: $CONFIG..."
    python3 src/evaluate.py \
        --model "$QUANT_DIR" \
        --config-name "$CONFIG" \
        --data-dir "$DATA_DIR" \
        --output-dir "$RESULTS_DIR" \
        --tp "$TP_SIZE"
    echo "  Done: $CONFIG"
    echo ""
done
echo "All evaluations complete."
echo ""

# --- Phase 4: Generate comparison report ---
echo "[Phase 4] Generating comparison report..."
python3 src/compare.py \
    --results-dir "$RESULTS_DIR" \
    --output "$RESULTS_DIR/summary.md" \
    --baseline baseline

echo ""
echo "========================================="
echo "  Experiment complete!"
echo "  Results: $RESULTS_DIR/summary.md"
echo "========================================="
