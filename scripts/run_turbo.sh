#!/usr/bin/env bash
# TurboQuant-inspired experiment runner.
# Run as: bash scripts/run_turbo.sh [--tp 8] [--configs "rotation-int4 ternary"]
set -euo pipefail

# --- Defaults ---
TP_SIZE=8
MODEL_DIR="./models/Qwen3-32B"
DATA_DIR="./data"
RESULTS_DIR="./results"
OUTPUT_DIR="./models/quantized"
CONFIGS="rotation-int4 rotation-boundary int2-residual int2r-boundary ternary ternary-boundary"
SKIP_QUANT=false
REPORT_ONLY=false

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --tp) TP_SIZE="$2"; shift 2 ;;
        --configs) CONFIGS="$2"; shift 2 ;;
        --model) MODEL_DIR="$2"; shift 2 ;;
        --skip-quant) SKIP_QUANT=true; shift ;;
        --report-only) REPORT_ONLY=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "========================================="
echo "  TurboQuant-Inspired Experiments"
echo "========================================="
echo "Model:    $MODEL_DIR"
echo "TP size:  $TP_SIZE"
echo "Configs:  $CONFIGS"
echo "========================================="
echo ""

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate qnt

if [ "$REPORT_ONLY" = true ]; then
    echo "Generating report only..."
    python3 src/eval_turbo.py --report-only --output-dir "$RESULTS_DIR"
    exit 0
fi

# --- Phase 1: Quantize ---
if [ "$SKIP_QUANT" = false ]; then
    echo "[Phase 1] Quantizing models..."
    for CONFIG in $CONFIGS; do
        QUANT_DIR="${OUTPUT_DIR}/${CONFIG}"
        if [ -d "$QUANT_DIR" ] && [ -f "$QUANT_DIR/quant_config.json" ]; then
            echo "  Skipping $CONFIG (already quantized)"
            continue
        fi
        echo "  Quantizing: $CONFIG..."
        python3 src/quantize_turbo.py \
            --model "$MODEL_DIR" \
            --config "$CONFIG" \
            --output-dir "$OUTPUT_DIR"
        echo "  Done: $CONFIG"
    done
    echo "All models quantized."
    echo ""
fi

# --- Phase 2: Evaluate ---
echo "[Phase 2] Evaluating quantized models..."
python3 src/eval_turbo.py \
    --configs $CONFIGS \
    --model-dir "$OUTPUT_DIR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$RESULTS_DIR" \
    --tp "$TP_SIZE"

echo ""
echo "========================================="
echo "  Experiment complete!"
echo "  Results: $RESULTS_DIR/turbo_summary.md"
echo "========================================="
