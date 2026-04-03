"""
Binary (1-bit) + INT4 mixed quantization with FP16 boundary layers.

Standalone quantization script for the binary mixing experiment.
Boundary layers (0, 1, 62, 63) stay at FP16. Middle layers (2-61)
get a mix of INT4 and 1-bit (binary: +scale or -scale per channel).

4 mixing strategies:
  binary-v1: 75% INT4, 25% 1-bit (every 4th middle layer is 1-bit)
  binary-v2: 50/50 alternating INT4 and 1-bit
  binary-v3: 25% INT4, 75% 1-bit (every 4th middle layer is INT4)
  binary-v4: gradient (INT4 near boundaries, 1-bit in the center)
"""

import gc
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

NUM_LAYERS = 64
BOUNDARY_LAYERS = {0, 1, 62, 63}
MIDDLE_LAYERS = list(range(2, 62))  # 60 layers

# Weight names within each transformer layer
ALL_WEIGHT_PATHS = {
    "q_proj": "self_attn.q_proj.weight",
    "k_proj": "self_attn.k_proj.weight",
    "v_proj": "self_attn.v_proj.weight",
    "o_proj": "self_attn.o_proj.weight",
    "gate_proj": "mlp.gate_proj.weight",
    "up_proj": "mlp.up_proj.weight",
    "down_proj": "mlp.down_proj.weight",
}


class LayerPrec(Enum):
    FP16 = "fp16"
    INT4 = "int4"
    BINARY = "1bit"


# ---------------------------------------------------------------------------
# Mixing strategies: return set of layer indices that should be 1-bit
# ---------------------------------------------------------------------------

def _binary_layers_v1() -> set[int]:
    """75% INT4, 25% 1-bit: every 4th middle layer is 1-bit."""
    return {MIDDLE_LAYERS[i] for i in range(0, len(MIDDLE_LAYERS), 4)}


def _binary_layers_v2() -> set[int]:
    """50/50: alternating INT4 and 1-bit."""
    return {MIDDLE_LAYERS[i] for i in range(1, len(MIDDLE_LAYERS), 2)}


def _binary_layers_v3() -> set[int]:
    """25% INT4, 75% 1-bit: every 4th middle layer is INT4, rest 1-bit."""
    int4_indices = {MIDDLE_LAYERS[i] for i in range(0, len(MIDDLE_LAYERS), 4)}
    return set(MIDDLE_LAYERS) - int4_indices


def _binary_layers_v4() -> set[int]:
    """Gradient: INT4 near boundaries, 1-bit in the center.

    Layers closest to boundaries stay INT4, layers in the dead center
    go 1-bit. Uses distance from nearest boundary to decide.
    """
    n = len(MIDDLE_LAYERS)
    mid = n // 2
    # Inner 60% of middle layers are 1-bit, outer 40% stay INT4
    cutoff = int(n * 0.20)  # 20% from each side stays INT4
    binary_indices = set()
    for i, layer_idx in enumerate(MIDDLE_LAYERS):
        if cutoff <= i < n - cutoff:
            binary_indices.add(layer_idx)
    return binary_indices


CONFIGS = {
    "binary-v1": {
        "description": "75% INT4, 25% 1-bit (every 4th middle layer is 1-bit)",
        "binary_layers_fn": _binary_layers_v1,
    },
    "binary-v2": {
        "description": "50/50 alternating INT4 and 1-bit",
        "binary_layers_fn": _binary_layers_v2,
    },
    "binary-v3": {
        "description": "25% INT4, 75% 1-bit (every 4th middle layer is INT4)",
        "binary_layers_fn": _binary_layers_v3,
    },
    "binary-v4": {
        "description": "Gradient: INT4 near boundaries, 1-bit in center",
        "binary_layers_fn": _binary_layers_v4,
    },
}


def get_layer_precision(config_name: str, layer_idx: int) -> LayerPrec:
    """Return the precision for a given layer index under a config."""
    if layer_idx in BOUNDARY_LAYERS:
        return LayerPrec.FP16
    binary_layers = CONFIGS[config_name]["binary_layers_fn"]()
    if layer_idx in binary_layers:
        return LayerPrec.BINARY
    return LayerPrec.INT4


def print_layer_map(config_name: str) -> str:
    """Print a visual map of layer precisions."""
    cfg = CONFIGS[config_name]
    binary_layers = cfg["binary_layers_fn"]()
    n_binary = len(binary_layers)
    n_int4 = len(MIDDLE_LAYERS) - n_binary
    n_fp16 = len(BOUNDARY_LAYERS)

    lines = [
        f"{config_name}: {cfg['description']}",
        f"  FP16: {n_fp16} layers (boundary)",
        f"  INT4: {n_int4} layers ({n_int4/60*100:.0f}% of middle)",
        f"  1-bit: {n_binary} layers ({n_binary/60*100:.0f}% of middle)",
        f"  Est. avg bits/param: {_est_avg_bits(config_name):.2f}",
        "",
        "  Layer map (F=FP16, 4=INT4, 1=1-bit):",
        "  ",
    ]

    row = "  "
    for i in range(NUM_LAYERS):
        prec = get_layer_precision(config_name, i)
        if prec == LayerPrec.FP16:
            row += "F"
        elif prec == LayerPrec.INT4:
            row += "4"
        else:
            row += "1"
        if (i + 1) % 32 == 0:
            lines.append(row)
            row = "  "
    if len(row.strip()) > 0:
        lines.append(row)

    return "\n".join(lines)


def _est_avg_bits(config_name: str) -> float:
    """Estimate average bits per parameter."""
    binary_layers = CONFIGS[config_name]["binary_layers_fn"]()
    total_bits = 0.0
    for i in range(NUM_LAYERS):
        if i in BOUNDARY_LAYERS:
            total_bits += 16.0
        elif i in binary_layers:
            # 1-bit weight + FP16 scale per output channel
            # Effective ~1.01 bits for large matrices, call it 1.0
            total_bits += 1.0
        else:
            total_bits += 4.0
    return total_bits / NUM_LAYERS


# ---------------------------------------------------------------------------
# Quantization functions
# ---------------------------------------------------------------------------

def quantize_binary(weight: torch.Tensor) -> torch.Tensor:
    """
    Binary (1-bit) quantization: each weight becomes +scale or -scale.

    Per output channel (dim=0): compute the mean absolute value as scale,
    then each weight is sign(w) * scale. This is the simplest binary
    quantization that preserves the magnitude distribution per channel.

    Returns dequantized FP16 tensor (simulated binary weights).
    """
    # Per-channel scale = mean of absolute values
    scale = weight.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    # Binary: just the sign
    signs = weight.sign()
    # Dequantize: sign * scale
    return (signs * scale).to(weight.dtype)


def quantize_int4(weight: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """
    Symmetric group-wise INT4 quantization (quantize + dequantize).

    Returns dequantized FP16 tensor (simulated INT4 weights).
    """
    rows, cols = weight.shape
    orig_cols = cols

    # Pad to multiple of group_size
    if cols % group_size != 0:
        pad = group_size - (cols % group_size)
        weight = torch.nn.functional.pad(weight, (0, pad))
        cols = weight.shape[1]

    n_groups = cols // group_size
    w_grouped = weight.reshape(rows, n_groups, group_size)

    amax = w_grouped.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
    scale = amax / 7.0
    quantized = torch.round(w_grouped / scale).clamp(-8, 7)
    dequantized = (quantized * scale).reshape(rows, cols)

    return dequantized[:, :orig_cols].to(weight.dtype)


def quantize_layer_weight(
    weight: torch.Tensor, precision: LayerPrec, group_size: int = 128
) -> torch.Tensor:
    """Apply quantization based on precision level."""
    if precision == LayerPrec.FP16:
        return weight
    elif precision == LayerPrec.INT4:
        return quantize_int4(weight, group_size)
    elif precision == LayerPrec.BINARY:
        return quantize_binary(weight)
    else:
        raise ValueError(f"Unknown precision: {precision}")


# ---------------------------------------------------------------------------
# Main quantization pipeline
# ---------------------------------------------------------------------------

def quantize_model(
    model_path: str,
    config_name: str,
    output_path: str,
) -> dict:
    """
    Apply binary/INT4 mixed quantization to a model.

    Args:
        model_path: Path to HuggingFace model directory
        config_name: One of binary-v1, binary-v2, binary-v3, binary-v4
        output_path: Where to save the quantized model

    Returns:
        Dict with quantization stats
    """
    if config_name not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Config: {config_name}")
    logger.info(f"\n{print_layer_map(config_name)}")
    logger.info(f"Model: {model_path} -> {output_path}")

    start_time = time.time()
    stats = {
        "config": config_name,
        "description": CONFIGS[config_name]["description"],
        "est_avg_bits": _est_avg_bits(config_name),
        "layers": [],
    }

    # Load model on CPU
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize layer by layer
    for layer_idx in tqdm(range(NUM_LAYERS), desc="Quantizing layers"):
        precision = get_layer_precision(config_name, layer_idx)
        layer_stats = {
            "layer": layer_idx,
            "precision": precision.value,
            "projections": {},
        }

        layer_module = model.model.layers[layer_idx]

        for proj_name, weight_path in ALL_WEIGHT_PATHS.items():
            parts = weight_path.split(".")
            module = layer_module
            for part in parts[:-1]:
                module = getattr(module, part)
            param_name = parts[-1]
            weight = getattr(module, param_name)

            if precision != LayerPrec.FP16:
                original_norm = weight.data.float().norm().item()
                weight.data = quantize_layer_weight(weight.data, precision)
                quantized_norm = weight.data.float().norm().item()

                layer_stats["projections"][proj_name] = {
                    "original_norm": original_norm,
                    "quantized_norm": quantized_norm,
                    "relative_error": abs(quantized_norm - original_norm)
                    / (original_norm + 1e-8),
                }

        stats["layers"].append(layer_stats)

    # Save
    logger.info(f"Saving to {output_path}...")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    elapsed = time.time() - start_time
    stats["total_time_seconds"] = elapsed

    meta_path = output_dir / "quant_config.json"
    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Done in {elapsed:.1f}s")

    del model
    gc.collect()

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Binary + INT4 mixed quantization with FP16 boundaries"
    )
    parser.add_argument("--model", required=True, help="Path to HF model")
    parser.add_argument(
        "--config", required=True,
        choices=list(CONFIGS.keys()) + ["all"],
        help="Config name or 'all' to run all 4",
    )
    parser.add_argument("--output-dir", default="./models/quantized",
                        help="Base output directory")
    parser.add_argument("--list-configs", action="store_true",
                        help="Print all configs and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.list_configs:
        for name in CONFIGS:
            print(print_layer_map(name))
            print()
        return

    configs_to_run = list(CONFIGS.keys()) if args.config == "all" else [args.config]

    for config_name in configs_to_run:
        output_path = f"{args.output_dir}/{config_name}"
        if Path(output_path).exists() and (Path(output_path) / "quant_config.json").exists():
            logger.info(f"Skipping {config_name} (already exists)")
            continue
        stats = quantize_model(args.model, config_name, output_path)
        print(f"\n{config_name}: done in {stats['total_time_seconds']:.1f}s")
        print(f"  Est. avg bits/param: {stats['est_avg_bits']:.2f}")
        print()


if __name__ == "__main__":
    main()
