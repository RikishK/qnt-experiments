"""
HumanEval and MBPP evaluation harness.

Generates code completions via vLLM with Qwen3 chat template
(enable_thinking=False) and executes them against test cases
to compute pass@1 scores. Also tracks exact match rate vs a baseline.
"""

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_from_disk
from tqdm import tqdm

from inference import InferenceConfig, VLLMInference

logger = logging.getLogger(__name__)

EXECUTION_TIMEOUT = 10  # seconds per test case


@dataclass
class EvalResult:
    """Evaluation result for a single problem."""
    task_id: str
    prompt: str
    generated_code: str
    passed: bool
    error: str | None = None
    execution_time: float = 0.0


@dataclass
class BenchmarkResult:
    """Aggregate results for a benchmark."""
    benchmark: str
    config_name: str
    model_path: str
    total_problems: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    pass_at_1: float = 0.0
    exact_match_rate: float | None = None  # vs baseline, if provided
    results: list[EvalResult] = field(default_factory=list)
    total_time: float = 0.0

    def summary(self) -> str:
        return (
            f"{self.benchmark} | {self.config_name}\n"
            f"  pass@1: {self.pass_at_1:.1%} "
            f"({self.passed}/{self.total_problems})\n"
            f"  errors: {self.errors}\n"
            f"  time: {self.total_time:.1f}s"
            + (f"\n  exact_match: {self.exact_match_rate:.1%}"
               if self.exact_match_rate is not None else "")
        )


def _execute_code_safely(code: str, test_code: str, timeout: int = EXECUTION_TIMEOUT) -> tuple[bool, str | None]:
    """
    Execute generated code + test case in a subprocess with timeout.

    Returns (passed, error_message).
    """
    full_code = code + "\n" + test_code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, None
        return False, result.stderr[:500]
    except subprocess.TimeoutExpired:
        return False, f"Timeout ({timeout}s)"
    except Exception as e:
        return False, str(e)[:500]
    finally:
        os.unlink(tmp_path)


def _extract_code_from_response(response: str) -> str:
    """
    Extract Python code from model response.

    Handles:
    - Code wrapped in ```python ... ``` blocks
    - Code wrapped in ``` ... ``` blocks
    - Raw code (no markdown fencing)
    """
    # Try to find ```python ... ``` block first
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try generic ``` ... ``` block
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # No code blocks found -- return the raw response
    # (strip leading/trailing whitespace and common preamble)
    return response.strip()


# ---------------------------------------------------------------------------
# HumanEval
# ---------------------------------------------------------------------------

def load_humaneval(data_dir: str = "./data/humaneval") -> list[dict]:
    """Load HumanEval dataset."""
    ds = load_from_disk(data_dir)
    problems = []
    for item in ds:
        problems.append({
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "canonical_solution": item["canonical_solution"],
            "test": item["test"],
            "entry_point": item["entry_point"],
        })
    return problems


def _build_humaneval_prompt(problem: dict) -> str:
    """
    Build a chat-style prompt for HumanEval.

    Includes the function signature and asks the model to complete it.
    """
    return (
        f"Complete the following Python function. "
        f"Return ONLY the complete function implementation inside a "
        f"```python``` code block. Do not include any explanation.\n\n"
        f"{problem['prompt']}"
    )


def eval_humaneval(
    engine: VLLMInference,
    config_name: str,
    data_dir: str = "./data/humaneval",
    max_tokens: int = 1024,
) -> BenchmarkResult:
    """Run HumanEval evaluation."""
    problems = load_humaneval(data_dir)
    logger.info(f"Running HumanEval: {len(problems)} problems")

    # Generate completions using chat-formatted prompts
    prompts = [_build_humaneval_prompt(p) for p in problems]
    start = time.time()
    gen_results = engine.generate(prompts, max_tokens=max_tokens)
    gen_time = time.time() - start

    # Execute and check
    result = BenchmarkResult(
        benchmark="humaneval",
        config_name=config_name,
        model_path=engine.config.model_path,
        total_problems=len(problems),
        total_time=gen_time,
    )

    for problem, gen in tqdm(
        zip(problems, gen_results), total=len(problems), desc="Executing HumanEval"
    ):
        # Extract code from model response
        raw_code = _extract_code_from_response(gen.output)

        # The model should return the full function. If it only returned
        # the body, prepend the signature.
        if not raw_code.startswith("def "):
            full_code = problem["prompt"] + raw_code
        else:
            full_code = raw_code

        # Build test harness
        test_code = problem["test"] + f"\ncheck({problem['entry_point']})\n"

        passed, error = _execute_code_safely(full_code, test_code)

        eval_r = EvalResult(
            task_id=problem["task_id"],
            prompt=problem["prompt"],
            generated_code=raw_code,
            passed=passed,
            error=error,
        )
        result.results.append(eval_r)

        if passed:
            result.passed += 1
        elif error:
            result.errors += 1
        else:
            result.failed += 1

    result.pass_at_1 = result.passed / result.total_problems
    logger.info(f"HumanEval pass@1: {result.pass_at_1:.1%}")
    return result


# ---------------------------------------------------------------------------
# MBPP
# ---------------------------------------------------------------------------

def load_mbpp(data_dir: str = "./data/mbpp") -> list[dict]:
    """Load MBPP sanitized dataset."""
    ds = load_from_disk(data_dir)
    problems = []
    for item in ds:
        problems.append({
            "task_id": str(item["task_id"]),
            "prompt": item["prompt"],
            "code": item["code"],
            "test_list": item["test_list"],
        })
    return problems


def _build_mbpp_prompt(problem: dict) -> str:
    """
    Build a chat-style prompt for an MBPP problem.

    Includes the task description and example test cases.
    """
    test_examples = "\n".join(problem["test_list"][:2])  # Show first 2 tests
    return (
        f"Write a Python function to solve the following task. "
        f"Return ONLY the function implementation inside a "
        f"```python``` code block. Do not include any explanation.\n\n"
        f"Task: {problem['prompt']}\n\n"
        f"Example test cases:\n{test_examples}"
    )


def eval_mbpp(
    engine: VLLMInference,
    config_name: str,
    data_dir: str = "./data/mbpp",
    max_tokens: int = 1024,
) -> BenchmarkResult:
    """Run MBPP evaluation."""
    problems = load_mbpp(data_dir)
    logger.info(f"Running MBPP: {len(problems)} problems")

    # Generate completions using chat-formatted prompts
    prompts = [_build_mbpp_prompt(p) for p in problems]
    start = time.time()
    gen_results = engine.generate(prompts, max_tokens=max_tokens)
    gen_time = time.time() - start

    # Execute and check
    result = BenchmarkResult(
        benchmark="mbpp",
        config_name=config_name,
        model_path=engine.config.model_path,
        total_problems=len(problems),
        total_time=gen_time,
    )

    for problem, gen in tqdm(
        zip(problems, gen_results), total=len(problems), desc="Executing MBPP"
    ):
        raw_code = _extract_code_from_response(gen.output)
        test_code = "\n".join(problem["test_list"])

        passed, error = _execute_code_safely(raw_code, test_code)

        eval_r = EvalResult(
            task_id=problem["task_id"],
            prompt=_build_mbpp_prompt(problem),
            generated_code=raw_code,
            passed=passed,
            error=error,
        )
        result.results.append(eval_r)

        if passed:
            result.passed += 1
        elif error:
            result.errors += 1
        else:
            result.failed += 1

    result.pass_at_1 = result.passed / result.total_problems
    logger.info(f"MBPP pass@1: {result.pass_at_1:.1%}")
    return result


# ---------------------------------------------------------------------------
# Exact match comparison
# ---------------------------------------------------------------------------

def compute_exact_match(
    baseline_results: list[EvalResult],
    quantized_results: list[EvalResult],
) -> float:
    """
    Compute exact match rate: fraction of problems where the quantized
    model produces character-identical code to the baseline.
    """
    assert len(baseline_results) == len(quantized_results)
    matches = sum(
        1 for b, q in zip(baseline_results, quantized_results)
        if b.generated_code.strip() == q.generated_code.strip()
    )
    return matches / len(baseline_results)


# ---------------------------------------------------------------------------
# Save / load results
# ---------------------------------------------------------------------------

def save_results(result: BenchmarkResult, output_dir: str) -> str:
    """Save benchmark results to JSON."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    filename = f"{result.benchmark}_{result.config_name}.json"
    filepath = path / filename

    data = {
        "benchmark": result.benchmark,
        "config_name": result.config_name,
        "model_path": result.model_path,
        "total_problems": result.total_problems,
        "passed": result.passed,
        "failed": result.failed,
        "errors": result.errors,
        "pass_at_1": result.pass_at_1,
        "exact_match_rate": result.exact_match_rate,
        "total_time": result.total_time,
        "results": [
            {
                "task_id": r.task_id,
                "passed": r.passed,
                "generated_code": r.generated_code,
                "error": r.error,
            }
            for r in result.results
        ],
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Results saved to {filepath}")
    return str(filepath)


def load_results(filepath: str) -> BenchmarkResult:
    """Load benchmark results from JSON."""
    with open(filepath) as f:
        data = json.load(f)

    result = BenchmarkResult(
        benchmark=data["benchmark"],
        config_name=data["config_name"],
        model_path=data["model_path"],
        total_problems=data["total_problems"],
        passed=data["passed"],
        failed=data["failed"],
        errors=data["errors"],
        pass_at_1=data["pass_at_1"],
        exact_match_rate=data.get("exact_match_rate"),
        total_time=data["total_time"],
    )

    for r in data["results"]:
        result.results.append(EvalResult(
            task_id=r["task_id"],
            prompt="",
            generated_code=r["generated_code"],
            passed=r["passed"],
            error=r.get("error"),
        ))

    return result


# ---------------------------------------------------------------------------
# Main: run full evaluation for a single config
# ---------------------------------------------------------------------------

def run_evaluation(
    model_path: str,
    config_name: str,
    data_dir: str = "./data",
    output_dir: str = "./results",
    tp_size: int = 8,
    benchmarks: list[str] | None = None,
) -> dict[str, BenchmarkResult]:
    """
    Run full evaluation pipeline for a model config.

    Args:
        model_path: Path to (quantized) model
        config_name: Name for this config in results
        data_dir: Directory containing humaneval/ and mbpp/ subdirs
        output_dir: Where to save results
        tp_size: Tensor parallelism size
        benchmarks: Which benchmarks to run (default: both)

    Returns:
        Dict mapping benchmark name to BenchmarkResult
    """
    if benchmarks is None:
        benchmarks = ["humaneval", "mbpp"]

    # Initialize vLLM
    inf_config = InferenceConfig(
        model_path=model_path,
        tensor_parallel_size=tp_size,
    )
    engine = VLLMInference(inf_config)
    engine.load()

    results = {}

    if "humaneval" in benchmarks:
        he_result = eval_humaneval(
            engine, config_name,
            data_dir=os.path.join(data_dir, "humaneval"),
        )
        save_results(he_result, output_dir)
        results["humaneval"] = he_result
        print(he_result.summary())
        print()

    if "mbpp" in benchmarks:
        mbpp_result = eval_mbpp(
            engine, config_name,
            data_dir=os.path.join(data_dir, "mbpp"),
        )
        save_results(mbpp_result, output_dir)
        results["mbpp"] = mbpp_result
        print(mbpp_result.summary())
        print()

    engine.unload()
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model on code benchmarks")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--config-name", required=True, help="Config name for results")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--output-dir", default="./results", help="Output directory")
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["humaneval", "mbpp"],
        choices=["humaneval", "mbpp"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    run_evaluation(
        args.model, args.config_name, args.data_dir,
        args.output_dir, args.tp, args.benchmarks,
    )


if __name__ == "__main__":
    main()
