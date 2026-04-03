"""
Quantization configuration definitions for Boundary V experiments.

Each config specifies per-layer, per-projection precision policies
for the Qwen3-32B model (64 layers, pure attention).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Precision(Enum):
    """Weight precision levels."""
    FP16 = "fp16"      # 16 bits - full precision baseline
    INT8 = "int8"      # 8 bits - symmetric per-channel quantization
    INT4 = "int4"      # 4 bits - group quantization (group_size=128)
    NF4 = "nf4"        # 4 bits - normal float (bitsandbytes)


@dataclass
class LayerPolicy:
    """Precision policy for a single transformer layer's projections."""
    q_proj: Precision = Precision.FP16
    k_proj: Precision = Precision.FP16
    v_proj: Precision = Precision.FP16
    o_proj: Precision = Precision.FP16
    gate_proj: Precision = Precision.FP16  # MLP
    up_proj: Precision = Precision.FP16    # MLP
    down_proj: Precision = Precision.FP16  # MLP


@dataclass
class QuantConfig:
    """Full quantization config for the model."""
    name: str
    description: str
    num_layers: int = 64  # Qwen3-32B
    boundary_width: int = 2  # Number of boundary layers on each side
    boundary_policy: LayerPolicy = field(default_factory=LayerPolicy)
    middle_policy: LayerPolicy = field(default_factory=LayerPolicy)
    # Calibration settings (for GPTQ-style quant)
    calibration_samples: int = 128
    calibration_seq_len: int = 2048
    group_size: int = 128  # For INT4 group quantization

    def get_layer_policy(self, layer_idx: int) -> LayerPolicy:
        """Return the precision policy for a given layer index."""
        is_boundary = (
            layer_idx < self.boundary_width
            or layer_idx >= self.num_layers - self.boundary_width
        )
        return self.boundary_policy if is_boundary else self.middle_policy

    def effective_bits_per_param(self) -> dict[str, float]:
        """Calculate effective bits per parameter for each projection type."""
        bits_map = {
            Precision.FP16: 16.0,
            Precision.INT8: 8.0,
            Precision.INT4: 4.0,
            Precision.NF4: 4.0,
        }
        n_boundary = self.boundary_width * 2
        n_middle = self.num_layers - n_boundary

        results = {}
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]:
            b_bits = bits_map[getattr(self.boundary_policy, proj)]
            m_bits = bits_map[getattr(self.middle_policy, proj)]
            avg = (n_boundary * b_bits + n_middle * m_bits) / self.num_layers
            results[proj] = avg
        return results

    def summary(self) -> str:
        """Human-readable config summary."""
        bits = self.effective_bits_per_param()
        avg_bits = sum(bits.values()) / len(bits)
        return (
            f"{self.name}: {self.description}\n"
            f"  Boundary (first/last {self.boundary_width}): "
            f"K={self.boundary_policy.k_proj.value} "
            f"V={self.boundary_policy.v_proj.value} "
            f"MLP={self.boundary_policy.gate_proj.value}\n"
            f"  Middle ({self.num_layers - self.boundary_width * 2} layers): "
            f"K={self.middle_policy.k_proj.value} "
            f"V={self.middle_policy.v_proj.value} "
            f"MLP={self.middle_policy.gate_proj.value}\n"
            f"  Avg bits/param: {avg_bits:.2f}"
        )


# ---------------------------------------------------------------------------
# Predefined experiment configs
# ---------------------------------------------------------------------------

def _uniform_policy(precision: Precision) -> LayerPolicy:
    return LayerPolicy(
        q_proj=precision, k_proj=precision, v_proj=precision,
        o_proj=precision, gate_proj=precision, up_proj=precision,
        down_proj=precision,
    )

CONFIGS: dict[str, QuantConfig] = {}


# Baseline: full FP16
CONFIGS["baseline"] = QuantConfig(
    name="baseline",
    description="Full FP16 precision (no quantization)",
    boundary_policy=_uniform_policy(Precision.FP16),
    middle_policy=_uniform_policy(Precision.FP16),
)

# Uniform INT8 -- standard 8-bit quantization everywhere
CONFIGS["uniform-int8"] = QuantConfig(
    name="uniform-int8",
    description="Uniform INT8 quantization across all layers",
    boundary_policy=_uniform_policy(Precision.INT8),
    middle_policy=_uniform_policy(Precision.INT8),
)

# Uniform INT4 -- standard 4-bit quantization everywhere
CONFIGS["uniform-int4"] = QuantConfig(
    name="uniform-int4",
    description="Uniform INT4 quantization across all layers",
    boundary_policy=_uniform_policy(Precision.INT4),
    middle_policy=_uniform_policy(Precision.INT4),
)

# Boundary V1: FP16 boundary, INT4 middle (simple boundary protection)
CONFIGS["boundary-v1"] = QuantConfig(
    name="boundary-v1",
    description="FP16 boundary layers, INT4 middle layers",
    boundary_policy=_uniform_policy(Precision.FP16),
    middle_policy=_uniform_policy(Precision.INT4),
)

# Boundary V2: INT8 boundary, INT4 middle (paper's q8_0 boundary approach)
CONFIGS["boundary-v2"] = QuantConfig(
    name="boundary-v2",
    description="INT8 boundary layers, INT4 middle layers (paper LA-V7 analog)",
    boundary_policy=_uniform_policy(Precision.INT8),
    middle_policy=_uniform_policy(Precision.INT4),
)

# Boundary V3: Asymmetric K/V -- K gets more protection than V
CONFIGS["boundary-v3"] = QuantConfig(
    name="boundary-v3",
    description="Asymmetric K/V: K always INT8, V compressed more aggressively",
    boundary_policy=LayerPolicy(
        q_proj=Precision.INT8, k_proj=Precision.INT8,
        v_proj=Precision.FP16, o_proj=Precision.INT8,
        gate_proj=Precision.INT8, up_proj=Precision.INT8,
        down_proj=Precision.INT8,
    ),
    middle_policy=LayerPolicy(
        q_proj=Precision.INT4, k_proj=Precision.INT8,
        v_proj=Precision.INT4, o_proj=Precision.INT4,
        gate_proj=Precision.INT4, up_proj=Precision.INT4,
        down_proj=Precision.INT4,
    ),
)

# Boundary V4: K-protection + boundary -- K stays INT8 everywhere,
# boundary layers FP16, middle INT4
CONFIGS["boundary-v4"] = QuantConfig(
    name="boundary-v4",
    description="K always INT8 + FP16 boundary layers, INT4 middle",
    boundary_policy=LayerPolicy(
        q_proj=Precision.FP16, k_proj=Precision.FP16,
        v_proj=Precision.FP16, o_proj=Precision.FP16,
        gate_proj=Precision.FP16, up_proj=Precision.FP16,
        down_proj=Precision.FP16,
    ),
    middle_policy=LayerPolicy(
        q_proj=Precision.INT4, k_proj=Precision.INT8,
        v_proj=Precision.INT4, o_proj=Precision.INT4,
        gate_proj=Precision.INT4, up_proj=Precision.INT4,
        down_proj=Precision.INT4,
    ),
)


def get_config(name: str) -> QuantConfig:
    """Get a config by name, raising KeyError if not found."""
    if name not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise KeyError(f"Unknown config '{name}'. Available: {available}")
    return CONFIGS[name]


def list_configs() -> None:
    """Print all available configs."""
    for config in CONFIGS.values():
        print(config.summary())
        print()


if __name__ == "__main__":
    list_configs()
