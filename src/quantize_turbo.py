"""
TurboQuant-inspired weight quantization experiments.

Three new quantization methods, all using FP16 boundary layers (0, 1, 62, 63):

1. Rotation-based INT4: Apply a random Hadamard rotation before INT4 quantization.
   TurboQuant's core insight is that random rotation spreads outliers evenly across
   coordinates, making naive per-element scalar quantization near-optimal. We test
   this on weight matrices: rotate -> INT4 -> dequantize -> inverse-rotate.

2. INT2 + 1-bit residual (~3 bits): Two-stage quantization inspired by TurboQuant's
   PolarQuant + QJL approach. Stage 1 uses INT2 (4 levels) to capture the bulk.
   Stage 2 applies 1-bit sign quantization to the residual (error from stage 1).
   The final dequantized weight = INT2_deq + sign(residual) * mean_abs(residual).

3. Ternary ({-1, 0, +1} * scale): Preserves near-zero weights instead of forcing
   them to +/- scale like binary does. Uses a threshold (0.7 * mean_abs) to decide
   which weights map to zero vs +/- 1.

Configs:
  rotation-int4:     Uniform rotation-based INT4 across all middle layers
  rotation-boundary: Rotation INT4 middle + INT8 boundary (like boundary-v2 but rotated)
  int2-residual:     Uniform INT2 + 1-bit residual across middle layers
  int2r-boundary:    INT2+residual middle + INT8 boundary
  ternary:           Uniform ternary across middle layers
  ternary-boundary:  Ternary middle + INT8 boundary
"""

import gc
import json
import logging
import math
import time
from enum import Enum
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

NUM_LAYERS = 64
BOUNDARY_LAYERS = {0, 1, 62, 63}
MIDDLE_LAYERS = list(range(2, 62))

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
    INT8 = "int8"
    ROTATION_INT4 = "rotation-int4"
    INT2_RESIDUAL = "int2+1bit"
    TERNARY = "ternary"


# ---------------------------------------------------------------------------
# Hadamard rotation utilities
# ---------------------------------------------------------------------------

def _hadamard_matrix(n: int) -> torch.Tensor:
    """Build a normalized Hadamard matrix of size n x n.

    Uses the Sylvester construction (recursive Kronecker product), so n must
    be a power of 2. The returned matrix H satisfies H @ H^T = I (orthogonal).
    """
    if n == 1:
        return torch.ones(1, 1)
    half = _hadamard_matrix(n // 2)
    top = torch.cat([half, half], dim=1)
    bottom = torch.cat([half, -half], dim=1)
    return torch.cat([top, bottom], dim=0) / math.sqrt(2)


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _apply_fast_hadamard(x: torch.Tensor) -> torch.Tensor:
    """Apply the Walsh-Hadamard transform along the last dimension in O(n log n).

    Input x has shape (..., n) where n is a power of 2. The transform uses
    the butterfly algorithm and is normalized by 1/sqrt(n).
    """
    n = x.shape[-1]
    h = 1
    while h < n:
        x = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a = x[..., 0, :].clone()
        b = x[..., 1, :].clone()
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        x = x.reshape(*x.shape[:-3], n)
        h *= 2
    return x / math.sqrt(n)


# ---------------------------------------------------------------------------
# Quantization functions
# ---------------------------------------------------------------------------

def quantize_rotation_int4(
    weight: torch.Tensor, group_size: int = 128, seed: int = 42
) -> torch.Tensor:
    """Rotation-based INT4 quantization (TurboQuant-inspired).

    1. Pad columns to next power of 2 (for Hadamard)
    2. Apply random sign flip + Hadamard rotation per row-block
    3. INT4 group-wise quantize (symmetric, 4-bit, group_size=128)
    4. Dequantize
    5. Inverse-rotate (Hadamard is self-inverse after sign flip)
    6. Remove padding

    The random sign flip + Hadamard "randomized Hadamard transform" (RHT)
    is equivalent to a random rotation that spreads outliers evenly,
    making uniform scalar quantization near-optimal per TurboQuant theory.
    """
    rows, cols = weight.shape
    orig_cols = cols

    # Pad to power of 2 for Hadamard
    padded_cols = _next_power_of_2(cols)
    if padded_cols != cols:
        weight = torch.nn.functional.pad(weight, (0, padded_cols - cols))
        cols = padded_cols

    # Random sign flip (diagonal of +/- 1)
    rng = torch.Generator()
    rng.manual_seed(seed)
    signs = torch.randint(0, 2, (cols,), generator=rng).float() * 2 - 1
    signs = signs.to(weight.dtype).to(weight.device)

    # Work in float32 to avoid FP16 overflow in butterfly additions.
    # Intermediate Hadamard values can be up to sqrt(n) * max(weight) before
    # normalization, which exceeds FP16 range for n >= 4096.
    orig_dtype = weight.dtype
    weight = weight.float()
    signs = signs.float()

    # Forward transform: sign flip then Hadamard
    rotated = weight * signs.unsqueeze(0)
    rotated = _apply_fast_hadamard(rotated)

    # INT4 group-wise quantization on the rotated weights
    if cols % group_size != 0:
        pad_g = group_size - (cols % group_size)
        rotated = torch.nn.functional.pad(rotated, (0, pad_g))
    r_rows, r_cols = rotated.shape
    n_groups = r_cols // group_size
    w_grouped = rotated.reshape(r_rows, n_groups, group_size)

    amax = w_grouped.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
    scale = amax / 7.0
    quantized = torch.round(w_grouped / scale).clamp(-8, 7)
    dequantized = (quantized * scale).reshape(r_rows, r_cols)
    dequantized = dequantized[:, :cols]  # remove group padding

    # Inverse transform: Hadamard then sign flip (Hadamard is self-inverse)
    restored = _apply_fast_hadamard(dequantized)
    restored = restored * signs.unsqueeze(0)

    return restored[:, :orig_cols].to(orig_dtype)


def quantize_int8(weight: torch.Tensor) -> torch.Tensor:
    """Symmetric per-channel INT8 quantization (quantize + dequantize)."""
    amax = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = amax / 127.0
    quantized = torch.round(weight / scale).clamp(-128, 127)
    return (quantized * scale).to(weight.dtype)


def quantize_int2_with_residual(weight: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Two-stage INT2 + 1-bit residual quantization (~3 bits per weight).

    Stage 1: INT2 symmetric group-wise quantization (4 levels: -1, 0, +1
    scaled, roughly {-1.5, -0.5, +0.5, +1.5} * scale). Captures the bulk
    of the weight distribution.

    Stage 2: 1-bit sign quantization on the residual (error from stage 1).
    Each residual element becomes sign(r) * mean_abs(r) per group. This
    corrects the quantization bias, inspired by TurboQuant's QJL stage.

    Total: 2 bits (INT2) + 1 bit (sign) = 3 bits per weight element,
    plus scales (amortized over group_size).
    """
    rows, cols = weight.shape
    orig_cols = cols

    if cols % group_size != 0:
        pad = group_size - (cols % group_size)
        weight = torch.nn.functional.pad(weight, (0, pad))
        cols = weight.shape[1]

    n_groups = cols // group_size
    w_grouped = weight.reshape(rows, n_groups, group_size)

    # Stage 1: INT2 -- 2-bit symmetric quantization
    # INT2 has range [-2, 1] or we use [-1.5, -0.5, 0.5, 1.5] mapped from [-2, -1, 0, 1]
    # Simpler: map to {-1, 0, 1} with scale (like ternary but from rounding to 2 bits)
    # Actually for true INT2: 4 values = {-2, -1, 0, 1} or unsigned {0, 1, 2, 3}
    # We use symmetric 2-bit: values in {-1, 0, 1} * scale (3 effective levels)
    # No -- true INT2 has 4 levels. Use symmetric: {-1.5, -0.5, 0.5, 1.5} * scale
    amax = w_grouped.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
    scale = amax / 1.5  # map max to 1.5, quantize to {-1.5, -0.5, 0.5, 1.5}
    normalized = w_grouped / scale
    # Round to nearest of {-1.5, -0.5, 0.5, 1.5}
    quantized = torch.round(normalized - 0.5) + 0.5  # shift to round to half-integers
    quantized = quantized.clamp(-1.5, 1.5)
    int2_deq = quantized * scale

    # Stage 2: 1-bit residual correction
    residual = w_grouped - int2_deq
    # Per-group: sign(residual) * mean_abs(residual)
    res_scale = residual.abs().mean(dim=2, keepdim=True).clamp(min=1e-8)
    res_signs = residual.sign()
    residual_deq = res_signs * res_scale

    # Combined output
    combined = (int2_deq + residual_deq).reshape(rows, cols)
    return combined[:, :orig_cols].to(weight.dtype)


def quantize_ternary(weight: torch.Tensor) -> torch.Tensor:
    """Ternary quantization: {-1, 0, +1} * scale per output channel.

    Uses a threshold to decide which weights map to zero vs +/- 1.
    Threshold = 0.7 * mean(|w|) per channel -- weights below threshold
    become 0, preserving the sparsity structure. Weights above threshold
    become +/- scale where scale = mean(|w| for |w| > threshold).

    This preserves near-zero weights instead of forcing them to +/- scale
    like binary does. Effective bit-width is ~1.58 bits (log2(3)).
    """
    # Per output channel (dim=1 = input features)
    mean_abs = weight.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    threshold = 0.7 * mean_abs

    # Ternary assignment
    mask_pos = weight > threshold
    mask_neg = weight < -threshold
    mask_zero = ~mask_pos & ~mask_neg

    # Scale = mean absolute value of non-zero weights
    abs_weight = weight.abs()
    non_zero_mask = mask_pos | mask_neg
    # Per-channel scale: mean of |w| where |w| > threshold
    scale = torch.where(
        non_zero_mask, abs_weight, torch.zeros_like(abs_weight)
    ).sum(dim=1, keepdim=True) / non_zero_mask.sum(dim=1, keepdim=True).clamp(min=1)

    # Build ternary output
    output = torch.zeros_like(weight)
    output[mask_pos] = scale.expand_as(weight)[mask_pos]
    output[mask_neg] = -scale.expand_as(weight)[mask_neg]

    return output.to(weight.dtype)


# ---------------------------------------------------------------------------
# Config system
# ---------------------------------------------------------------------------

CONFIGS = {
    "rotation-int4": {
        "description": "Hadamard-rotated INT4 uniform (all middle layers)",
        "boundary_prec": LayerPrec.FP16,
        "middle_prec": LayerPrec.ROTATION_INT4,
        "est_bits": lambda: (4 * 16 + 4.0 * 60) / 64,  # boundary FP16 counted as 16
    },
    "rotation-boundary": {
        "description": "Hadamard-rotated INT4 middle + INT8 boundary",
        "boundary_prec": LayerPrec.INT8,
        "middle_prec": LayerPrec.ROTATION_INT4,
        "est_bits": lambda: (4 * 8.0 + 4.0 * 60) / 64,
    },
    "int2-residual": {
        "description": "INT2 + 1-bit residual (~3 bits) uniform middle layers",
        "boundary_prec": LayerPrec.FP16,
        "middle_prec": LayerPrec.INT2_RESIDUAL,
        "est_bits": lambda: (4 * 16 + 3.0 * 60) / 64,
    },
    "int2r-boundary": {
        "description": "INT2 + 1-bit residual middle + INT8 boundary",
        "boundary_prec": LayerPrec.INT8,
        "middle_prec": LayerPrec.INT2_RESIDUAL,
        "est_bits": lambda: (4 * 8.0 + 3.0 * 60) / 64,
    },
    "ternary": {
        "description": "Ternary ({-1,0,+1} * scale) uniform middle layers",
        "boundary_prec": LayerPrec.FP16,
        "middle_prec": LayerPrec.TERNARY,
        "est_bits": lambda: (4 * 16 + 1.58 * 60) / 64,
    },
    "ternary-boundary": {
        "description": "Ternary middle + INT8 boundary",
        "boundary_prec": LayerPrec.INT8,
        "middle_prec": LayerPrec.TERNARY,
        "est_bits": lambda: (4 * 8.0 + 1.58 * 60) / 64,
    },
}


def get_layer_precision(config_name: str, layer_idx: int) -> LayerPrec:
    cfg = CONFIGS[config_name]
    if layer_idx in BOUNDARY_LAYERS:
        return cfg["boundary_prec"]
    return cfg["middle_prec"]


def print_layer_map(config_name: str) -> str:
    cfg = CONFIGS[config_name]
    est = cfg["est_bits"]()
    prec_chars = {
        LayerPrec.FP16: "F",
        LayerPrec.INT8: "8",
        LayerPrec.ROTATION_INT4: "R",
        LayerPrec.INT2_RESIDUAL: "2",
        LayerPrec.TERNARY: "T",
    }
    lines = [
        f"{config_name}: {cfg['description']}",
        f"  Boundary: {cfg['boundary_prec'].value}",
        f"  Middle: {cfg['middle_prec'].value}",
        f"  Est. avg bits/param: {est:.2f}",
        "",
        "  Layer map (F=FP16, 8=INT8, R=RotINT4, 2=INT2+1bit, T=Ternary):",
    ]
    row = "  "
    for i in range(NUM_LAYERS):
        prec = get_layer_precision(config_name, i)
        row += prec_chars[prec]
        if (i + 1) % 32 == 0:
            lines.append(row)
            row = "  "
    if row.strip():
        lines.append(row)
    return "\n".join(lines)


def quantize_weight(
    weight: torch.Tensor, precision: LayerPrec, group_size: int = 128
) -> torch.Tensor:
    if precision == LayerPrec.FP16:
        return weight
    if precision == LayerPrec.INT8:
        return quantize_int8(weight)
    if precision == LayerPrec.ROTATION_INT4:
        return quantize_rotation_int4(weight, group_size)
    if precision == LayerPrec.INT2_RESIDUAL:
        return quantize_int2_with_residual(weight, group_size)
    if precision == LayerPrec.TERNARY:
        return quantize_ternary(weight)
    raise ValueError(f"Unknown precision: {precision}")


# ---------------------------------------------------------------------------
# Main quantization pipeline
# ---------------------------------------------------------------------------

def quantize_model(
    model_path: str,
    config_name: str,
    output_path: str,
) -> dict:
    if config_name not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Config: {config_name}")
    logger.info(f"\n{print_layer_map(config_name)}")
    logger.info(f"Model: {model_path} -> {output_path}")

    start_time = time.time()
    cfg = CONFIGS[config_name]
    stats = {
        "config": config_name,
        "description": cfg["description"],
        "est_avg_bits": cfg["est_bits"](),
        "layers": [],
    }

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

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
                weight.data = quantize_weight(weight.data, precision)
                quantized_norm = weight.data.float().norm().item()

                layer_stats["projections"][proj_name] = {
                    "original_norm": original_norm,
                    "quantized_norm": quantized_norm,
                    "relative_error": abs(quantized_norm - original_norm)
                    / (original_norm + 1e-8),
                }

        stats["layers"].append(layer_stats)

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
        description="TurboQuant-inspired weight quantization experiments"
    )
    parser.add_argument("--model", required=True, help="Path to HF model")
    parser.add_argument(
        "--config", required=True,
        choices=list(CONFIGS.keys()) + ["all"],
        help="Config name or 'all' to run all",
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
