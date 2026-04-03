# Boundary V Quantization Experiments

Investigating **layer-aware mixed-precision quantization** on Qwen3-32B, based on the [Boundary V paper](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/layer-aware-v-compression.md).

**Core idea**: Boundary transformer layers (first 2 + last 2) are more sensitive to quantization error than middle layers. Protect them at higher precision, compress the rest aggressively.

## Quick Start

```bash
# 1. Setup EC2 (g6.48xlarge with 8x L4 GPUs)
bash scripts/setup_ec2.sh

# 2. Download model + datasets
bash scripts/download_model.sh

# 3. Run full experiment
bash scripts/run_all.sh
```

## Configs Tested

| Config | Boundary (4 layers) | Middle (60 layers) |
|--------|--------------------|--------------------|
| baseline | FP16 | FP16 |
| uniform-int8 | INT8 | INT8 |
| uniform-int4 | INT4 | INT4 |
| boundary-v1 | FP16 | INT4 |
| boundary-v2 | INT8 | INT4 |
| boundary-v3 | INT8 K, FP16 V | INT8 K, INT4 V |
| boundary-v4 | FP16 | INT8 K, INT4 rest |

## Evaluation

- **HumanEval** (164 problems) + **MBPP** (500 problems) at temperature=0
- Metrics: pass@1, exact match rate, perplexity, latency, throughput, VRAM

## Project Structure

```
scripts/       - EC2 setup, model download, experiment runner
src/           - Quantization, inference, evaluation, metrics, analysis
results/       - Output data and summary reports
```
