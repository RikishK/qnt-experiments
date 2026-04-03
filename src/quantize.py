"""
Layer-aware quantization engine for Boundary V experiments.

Applies per-layer, per-projection precision policies to a HuggingFace
transformer model and saves in a format vLLM can load.
"""

import gc
import json
import logging
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs import Precision, QuantConfig, get_config

logger = logging.getLogger(__name__)

# Projection weight name patterns in Qwen3 architecture
ATTENTION_PROJS = {
    "q_proj": "self_attn.q_proj.weight",
    "k_proj": "self_attn.k_proj.weight",
    "v_proj": "self_attn.v_proj.weight",
    "o_proj": "self_attn.o_proj.weight",
}
MLP_PROJS = {
    "gate_proj": "mlp.gate_proj.weight",
    "up_proj": "mlp.up_proj.weight",
    "down_proj": "mlp.down_proj.weight",
}
ALL_PROJS = {**ATTENTION_PROJS, **MLP_PROJS}


def quantize_tensor_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-channel INT8 quantization.

    Returns (quantized_int8, scales) where:
        dequantized = quantized_int8.float() * scales
    """
    # Per-output-channel (dim=0) symmetric quantization
    amax = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = amax / 127.0
    quantized = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
    return quantized, scale.squeeze(1)


def quantize_tensor_int4(
    weight: torch.Tensor, group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Symmetric group-wise INT4 quantization.

    Groups along the input dimension (dim=1). Returns
    (quantized_int8_packed, scales, zeros) -- we store INT4 as pairs
    packed into INT8 for compatibility.

    Returns (quantized_int8, scales, zeros) where each group of
    `group_size` elements shares a scale/zero.
    """
    rows, cols = weight.shape
    # Pad columns to multiple of group_size
    if cols % group_size != 0:
        pad = group_size - (cols % group_size)
        weight = torch.nn.functional.pad(weight, (0, pad))
        cols = weight.shape[1]

    n_groups = cols // group_size
    weight_grouped = weight.reshape(rows, n_groups, group_size)

    # Symmetric quantization per group
    amax = weight_grouped.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
    scale = amax / 7.0  # INT4 range: [-8, 7]
    quantized = torch.round(weight_grouped / scale).clamp(-8, 7).to(torch.int8)

    # Reshape back
    quantized = quantized.reshape(rows, cols)
    scale = scale.squeeze(2)  # (rows, n_groups)

    return quantized, scale


def dequantize_int8(
    quantized: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Dequantize INT8 back to float."""
    return quantized.float() * scale.unsqueeze(1)


def dequantize_int4(
    quantized: torch.Tensor, scale: torch.Tensor, group_size: int = 128
) -> torch.Tensor:
    """Dequantize INT4 back to float."""
    rows, cols = quantized.shape
    n_groups = cols // group_size
    q_grouped = quantized.reshape(rows, n_groups, group_size).float()
    return (q_grouped * scale.unsqueeze(2)).reshape(rows, cols)


def quantize_weight(
    weight: torch.Tensor,
    precision: Precision,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Quantize a weight tensor to the target precision and immediately
    dequantize back to FP16.

    This simulates quantized inference: the weights are stored at lower
    precision but computation happens in FP16. The quantization error
    is baked into the weights.
    """
    if precision == Precision.FP16:
        return weight

    if precision == Precision.INT8:
        q, scale = quantize_tensor_int8(weight)
        return dequantize_int8(q, scale).to(weight.dtype)

    if precision in (Precision.INT4, Precision.NF4):
        orig_cols = weight.shape[1]
        q, scale = quantize_tensor_int4(weight, group_size)
        deq = dequantize_int4(q, scale, group_size)
        return deq[:, :orig_cols].to(weight.dtype)

    raise ValueError(f"Unsupported precision: {precision}")


def quantize_model(
    model_path: str,
    config: QuantConfig,
    output_path: str,
    device: str = "cpu",
) -> dict:
    """
    Apply layer-aware quantization to a model.

    Loads the model, applies per-layer precision policies (quantize then
    dequantize to simulate quantized weights in FP16 format), and saves
    the result as safetensors that vLLM can load directly.

    Args:
        model_path: Path to HuggingFace model directory
        config: Quantization configuration
        output_path: Where to save the quantized model
        device: Device to use for quantization computation

    Returns:
        Dict with quantization stats (time, per-layer info)
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Quantizing with config: {config.name}")
    logger.info(f"Model: {model_path} -> {output_path}")
    logger.info(f"\n{config.summary()}")

    start_time = time.time()
    stats = {"config": config.name, "layers": []}

    # Load model -- use device_map="cpu" to avoid OOM, we quantize on CPU
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize layer by layer
    num_layers = config.num_layers
    for layer_idx in tqdm(range(num_layers), desc="Quantizing layers"):
        policy = config.get_layer_policy(layer_idx)
        layer_stats = {"layer": layer_idx, "projections": {}}

        layer_module = model.model.layers[layer_idx]

        for proj_name, weight_path in ALL_PROJS.items():
            precision = getattr(policy, proj_name)

            # Navigate to the weight tensor
            parts = weight_path.split(".")
            module = layer_module
            for part in parts[:-1]:
                module = getattr(module, part)
            param_name = parts[-1]
            weight = getattr(module, param_name)

            if precision != Precision.FP16:
                # Quantize + dequantize (simulated quantization)
                original_norm = weight.data.float().norm().item()
                weight.data = quantize_weight(
                    weight.data, precision, config.group_size
                )
                quantized_norm = weight.data.float().norm().item()

                layer_stats["projections"][proj_name] = {
                    "precision": precision.value,
                    "original_norm": original_norm,
                    "quantized_norm": quantized_norm,
                    "relative_error": abs(quantized_norm - original_norm)
                    / (original_norm + 1e-8),
                }
            else:
                layer_stats["projections"][proj_name] = {
                    "precision": "fp16",
                    "unchanged": True,
                }

        stats["layers"].append(layer_stats)

    # Save the quantized model
    logger.info(f"Saving quantized model to {output_path}...")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # Save quantization metadata
    elapsed = time.time() - start_time
    stats["total_time_seconds"] = elapsed
    stats["config_summary"] = config.summary()

    meta_path = output_dir / "quant_config.json"
    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Quantization complete in {elapsed:.1f}s")

    # Free memory
    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Layer-aware model quantization")
    parser.add_argument("--model", required=True, help="Path to HuggingFace model")
    parser.add_argument(
        "--config",
        required=True,
        help="Quantization config name (see configs.py)",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device for quantization")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = get_config(args.config)
    stats = quantize_model(args.model, config, args.output, args.device)

    print(f"\nDone. Config: {config.name}")
    print(f"Time: {stats['total_time_seconds']:.1f}s")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
