"""
vLLM inference wrapper for running generation across configs.

Handles tensor-parallel setup on 8x L4 GPUs and provides a
consistent interface for evaluation scripts.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from a single generation."""
    prompt: str
    output: str
    tokens_generated: int
    time_to_first_token: float  # seconds
    total_time: float  # seconds
    tokens_per_second: float


@dataclass
class InferenceConfig:
    """Configuration for vLLM inference."""
    model_path: str
    tensor_parallel_size: int = 8  # 8x L4 GPUs
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    dtype: str = "float16"
    temperature: float = 0.0  # Greedy decoding for deterministic comparison
    max_tokens: int = 1024
    seed: int = 42


class VLLMInference:
    """Wrapper around vLLM for batched inference."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.llm = None
        self.sampling_params = None

    def load(self) -> None:
        """Initialize the vLLM engine."""
        from vllm import LLM, SamplingParams

        logger.info(
            f"Loading model: {self.config.model_path} "
            f"(TP={self.config.tensor_parallel_size})"
        )
        start = time.time()

        self.llm = LLM(
            model=self.config.model_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            dtype=self.config.dtype,
            seed=self.config.seed,
            trust_remote_code=True,
        )

        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            # For temp=0 (greedy), these are ignored but set for clarity
            top_p=1.0,
            top_k=-1,
        )

        elapsed = time.time() - start
        logger.info(f"Model loaded in {elapsed:.1f}s")

    def generate(
        self,
        prompts: list[str],
        max_tokens: int | None = None,
    ) -> list[GenerationResult]:
        """
        Generate completions for a batch of prompts.

        Args:
            prompts: List of prompt strings
            max_tokens: Override default max_tokens if set

        Returns:
            List of GenerationResult objects
        """
        from vllm import SamplingParams

        if self.llm is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        params = self.sampling_params
        if max_tokens is not None:
            params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=max_tokens,
                top_p=1.0,
                top_k=-1,
            )

        logger.info(f"Generating {len(prompts)} completions...")
        start = time.time()

        outputs = self.llm.generate(prompts, params)

        total_time = time.time() - start
        results = []

        for i, output in enumerate(outputs):
            text = output.outputs[0].text
            n_tokens = len(output.outputs[0].token_ids)
            # vLLM doesn't expose TTFT directly in offline mode;
            # approximate with total_time / len(prompts) for now
            per_prompt_time = total_time / len(prompts)

            results.append(GenerationResult(
                prompt=prompts[i],
                output=text,
                tokens_generated=n_tokens,
                time_to_first_token=0.0,  # Not available in offline mode
                total_time=per_prompt_time,
                tokens_per_second=n_tokens / per_prompt_time if per_prompt_time > 0 else 0,
            ))

        logger.info(
            f"Generated {sum(r.tokens_generated for r in results)} tokens "
            f"in {total_time:.1f}s"
        )
        return results

    def get_gpu_memory_usage(self) -> list[dict]:
        """Get current GPU memory usage across all devices."""
        usage = []
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_mem / 1024**3
            usage.append({
                "gpu": i,
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "utilization_pct": round(allocated / total * 100, 1),
            })
        return usage

    def unload(self) -> None:
        """Release the model and free GPU memory."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Model unloaded, GPU memory freed.")


def run_inference(
    model_path: str,
    prompts: list[str],
    tp_size: int = 8,
    max_tokens: int = 1024,
    output_file: str | None = None,
) -> list[GenerationResult]:
    """
    Convenience function: load model, run inference, save results.

    Args:
        model_path: Path to model directory
        prompts: List of prompts
        tp_size: Tensor parallelism size
        max_tokens: Max tokens to generate per prompt
        output_file: Optional path to save results as JSONL

    Returns:
        List of GenerationResult
    """
    config = InferenceConfig(
        model_path=model_path,
        tensor_parallel_size=tp_size,
        max_tokens=max_tokens,
    )

    engine = VLLMInference(config)
    engine.load()

    # Log GPU memory after loading
    mem = engine.get_gpu_memory_usage()
    for gpu in mem:
        logger.info(
            f"GPU {gpu['gpu']}: {gpu['allocated_gb']:.1f}GB / "
            f"{gpu['total_gb']:.1f}GB ({gpu['utilization_pct']}%)"
        )

    results = engine.generate(prompts, max_tokens=max_tokens)
    engine.unload()

    # Optionally save to JSONL
    if output_file:
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in results:
                f.write(json.dumps({
                    "prompt": r.prompt,
                    "output": r.output,
                    "tokens_generated": r.tokens_generated,
                    "total_time": r.total_time,
                    "tokens_per_second": r.tokens_per_second,
                }) + "\n")
        logger.info(f"Results saved to {output_file}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run vLLM inference")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--prompt", help="Single prompt to test")
    parser.add_argument("--prompt-file", help="File with one prompt per line")
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", help="Output JSONL file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = ["def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n"]

    results = run_inference(
        args.model, prompts, args.tp, args.max_tokens, args.output
    )

    for r in results:
        print(f"--- Prompt ---\n{r.prompt}")
        print(f"--- Output ({r.tokens_generated} tokens, {r.tokens_per_second:.1f} t/s) ---")
        print(r.output)
        print()


if __name__ == "__main__":
    main()
