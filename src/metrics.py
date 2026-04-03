"""
Metrics collection: latency, throughput, VRAM, model size, perplexity.
"""

import gc
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a model config."""
    config_name: str
    model_path: str
    # Model size
    model_size_gb: float = 0.0
    num_parameters: int = 0
    # VRAM
    peak_vram_per_gpu_gb: list[float] = field(default_factory=list)
    total_vram_gb: float = 0.0
    # Latency
    time_to_first_token_ms: float = 0.0
    mean_decode_latency_ms: float = 0.0
    # Throughput
    tokens_per_second: float = 0.0
    prefill_tokens_per_second: float = 0.0
    # Perplexity
    perplexity: float | None = None

    def summary(self) -> str:
        lines = [
            f"Metrics: {self.config_name}",
            f"  Model size: {self.model_size_gb:.2f} GB",
            f"  Peak VRAM: {self.total_vram_gb:.2f} GB total",
            f"  Decode: {self.tokens_per_second:.1f} tok/s",
            f"  Decode latency: {self.mean_decode_latency_ms:.1f} ms/tok",
        ]
        if self.perplexity is not None:
            lines.append(f"  Perplexity: {self.perplexity:.3f}")
        return "\n".join(lines)


def measure_model_size(model_path: str) -> float:
    """Measure total safetensors size on disk in GB."""
    total = 0
    model_dir = Path(model_path)
    for f in model_dir.glob("*.safetensors"):
        total += f.stat().st_size
    # Also check for .bin files
    for f in model_dir.glob("*.bin"):
        total += f.stat().st_size
    return total / (1024 ** 3)


def measure_vram_usage() -> tuple[list[float], float]:
    """
    Measure current GPU VRAM allocation.

    Returns (per_gpu_gb_list, total_gb).
    """
    per_gpu = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.max_memory_allocated(i) / 1024**3
        per_gpu.append(round(allocated, 2))
    total = sum(per_gpu)
    return per_gpu, round(total, 2)


def measure_throughput(
    engine,  # VLLMInference instance
    num_prompts: int = 50,
    prompt_len: int = 128,
    max_tokens: int = 256,
) -> tuple[float, float]:
    """
    Measure decode throughput by running a batch of prompts.

    Returns (tokens_per_second, mean_decode_latency_ms).
    """
    # Generate simple prompts
    prompts = [
        f"Write a Python function that computes the {i}th value in a sequence:\n"
        for i in range(num_prompts)
    ]

    # Reset peak memory stats
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)

    start = time.time()
    results = engine.generate(prompts, max_tokens=max_tokens)
    total_time = time.time() - start

    total_tokens = sum(r.tokens_generated for r in results)
    tps = total_tokens / total_time if total_time > 0 else 0
    ms_per_tok = (total_time / total_tokens * 1000) if total_tokens > 0 else 0

    return tps, ms_per_tok


def measure_perplexity(
    engine,  # VLLMInference instance
    data_path: str = "./data/wikitext2",
    max_samples: int = 200,
    seq_len: int = 512,
) -> float:
    """
    Measure perplexity on wikitext-2.

    Uses vLLM's logprob support to compute perplexity efficiently.
    """
    from vllm import SamplingParams
    from datasets import load_from_disk

    ds = load_from_disk(data_path)

    # Concatenate all text and split into chunks
    all_text = "\n\n".join(
        t for t in ds["text"] if t.strip()
    )

    # Use the model's tokenizer to chunk properly
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        engine.config.model_path, trust_remote_code=True
    )

    tokens = tokenizer.encode(all_text)
    chunks = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        chunk_tokens = tokens[i : i + seq_len]
        chunks.append(tokenizer.decode(chunk_tokens))
        if len(chunks) >= max_samples:
            break

    if not chunks:
        logger.warning("No valid chunks for perplexity calculation")
        return float("inf")

    logger.info(f"Computing perplexity on {len(chunks)} chunks of {seq_len} tokens")

    # Use vLLM with logprobs to compute perplexity
    # We prompt with the text and ask for 1 token with logprobs
    # to get the log-likelihood of the sequence
    params = SamplingParams(
        temperature=0,
        max_tokens=1,
        prompt_logprobs=1,  # Get logprobs for prompt tokens
    )

    total_nll = 0.0
    total_tokens = 0

    # Process in batches
    batch_size = 20
    for batch_start in tqdm(
        range(0, len(chunks), batch_size), desc="Perplexity"
    ):
        batch = chunks[batch_start : batch_start + batch_size]
        outputs = engine.llm.generate(batch, params)

        for output in outputs:
            if output.prompt_logprobs is not None:
                for logprob_dict in output.prompt_logprobs:
                    if logprob_dict is not None:
                        # Get the logprob of the actual token
                        for token_id, logprob_obj in logprob_dict.items():
                            total_nll -= logprob_obj.logprob
                            total_tokens += 1

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    logger.info(f"Perplexity: {ppl:.3f} (over {total_tokens} tokens)")
    return ppl


def collect_all_metrics(
    engine,  # VLLMInference instance
    config_name: str,
    model_path: str,
    data_dir: str = "./data",
    run_perplexity: bool = True,
) -> PerformanceMetrics:
    """
    Collect all performance metrics for a model config.
    """
    metrics = PerformanceMetrics(
        config_name=config_name,
        model_path=model_path,
    )

    # Model size on disk
    metrics.model_size_gb = measure_model_size(model_path)
    logger.info(f"Model size: {metrics.model_size_gb:.2f} GB")

    # Throughput
    logger.info("Measuring throughput...")
    tps, ms_per_tok = measure_throughput(engine)
    metrics.tokens_per_second = tps
    metrics.mean_decode_latency_ms = ms_per_tok
    logger.info(f"Throughput: {tps:.1f} tok/s, {ms_per_tok:.1f} ms/tok")

    # VRAM (after throughput warmup)
    per_gpu, total = measure_vram_usage()
    metrics.peak_vram_per_gpu_gb = per_gpu
    metrics.total_vram_gb = total
    logger.info(f"Peak VRAM: {total:.2f} GB total")

    # Perplexity
    if run_perplexity:
        wikitext_path = os.path.join(data_dir, "wikitext2")
        if os.path.exists(wikitext_path):
            metrics.perplexity = measure_perplexity(
                engine, data_path=wikitext_path
            )
        else:
            logger.warning(f"Wikitext-2 not found at {wikitext_path}, skipping PPL")

    return metrics


def save_metrics(metrics: PerformanceMetrics, output_dir: str) -> str:
    """Save metrics to JSON."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    filepath = path / f"metrics_{metrics.config_name}.json"
    with open(filepath, "w") as f:
        json.dump({
            "config_name": metrics.config_name,
            "model_path": metrics.model_path,
            "model_size_gb": metrics.model_size_gb,
            "peak_vram_per_gpu_gb": metrics.peak_vram_per_gpu_gb,
            "total_vram_gb": metrics.total_vram_gb,
            "tokens_per_second": metrics.tokens_per_second,
            "mean_decode_latency_ms": metrics.mean_decode_latency_ms,
            "perplexity": metrics.perplexity,
        }, f, indent=2)

    logger.info(f"Metrics saved to {filepath}")
    return str(filepath)
