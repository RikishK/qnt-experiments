"""
Evaluation script for binary + INT4 mixed quantization experiments.

Standalone script that runs HumanEval and MBPP against the binary-v1
through binary-v4 quantized models, compares to the existing baseline
and uniform-int4 results, and generates a focused comparison report.
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

EXECUTION_TIMEOUT = 10

BINARY_CONFIGS = ["binary-v1", "binary-v2", "binary-v3", "binary-v4"]
# Reference configs to include in the comparison table
REFERENCE_CONFIGS = ["baseline", "uniform-int4", "boundary-v3"]


@dataclass
class EvalResult:
    task_id: str
    prompt: str
    generated_code: str
    passed: bool
    error: str | None = None


@dataclass
class BenchmarkResult:
    benchmark: str
    config_name: str
    model_path: str
    total_problems: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    pass_at_1: float = 0.0
    exact_match_rate: float | None = None
    results: list[EvalResult] = field(default_factory=list)
    total_time: float = 0.0


def _execute_code_safely(code: str, test_code: str, timeout: int = EXECUTION_TIMEOUT) -> tuple[bool, str | None]:
    full_code = code + "\n" + test_code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["python3", tmp_path], capture_output=True, text=True, timeout=timeout,
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
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    return response.strip()


# ---------------------------------------------------------------------------
# HumanEval
# ---------------------------------------------------------------------------

def eval_humaneval(engine: VLLMInference, config_name: str,
                   data_dir: str = "./data/humaneval", max_tokens: int = 1024) -> BenchmarkResult:
    ds = load_from_disk(data_dir)
    problems = [
        {"task_id": item["task_id"], "prompt": item["prompt"],
         "test": item["test"], "entry_point": item["entry_point"]}
        for item in ds
    ]
    logger.info(f"Running HumanEval ({config_name}): {len(problems)} problems")

    prompts = [
        f"Complete the following Python function. "
        f"Return ONLY the complete function implementation inside a "
        f"```python``` code block. Do not include any explanation.\n\n"
        f"{p['prompt']}"
        for p in problems
    ]

    start = time.time()
    gen_results = engine.generate(prompts, max_tokens=max_tokens)
    gen_time = time.time() - start

    result = BenchmarkResult(
        benchmark="humaneval", config_name=config_name,
        model_path=engine.config.model_path,
        total_problems=len(problems), total_time=gen_time,
    )

    for problem, gen in tqdm(zip(problems, gen_results), total=len(problems), desc="HumanEval"):
        raw_code = _extract_code_from_response(gen.output)
        full_code = raw_code if raw_code.startswith("def ") else problem["prompt"] + raw_code
        test_code = problem["test"] + f"\ncheck({problem['entry_point']})\n"
        passed, error = _execute_code_safely(full_code, test_code)
        result.results.append(EvalResult(
            task_id=problem["task_id"], prompt=problem["prompt"],
            generated_code=raw_code, passed=passed, error=error,
        ))
        if passed:
            result.passed += 1
        elif error:
            result.errors += 1
        else:
            result.failed += 1

    result.pass_at_1 = result.passed / result.total_problems
    logger.info(f"HumanEval pass@1: {result.pass_at_1:.1%} ({result.passed}/{result.total_problems})")
    return result


# ---------------------------------------------------------------------------
# MBPP
# ---------------------------------------------------------------------------

def eval_mbpp(engine: VLLMInference, config_name: str,
              data_dir: str = "./data/mbpp", max_tokens: int = 1024) -> BenchmarkResult:
    ds = load_from_disk(data_dir)
    problems = [
        {"task_id": str(item["task_id"]), "prompt": item["prompt"],
         "code": item["code"], "test_list": item["test_list"]}
        for item in ds
    ]
    logger.info(f"Running MBPP ({config_name}): {len(problems)} problems")

    prompts = [
        f"Write a Python function to solve the following task. "
        f"Return ONLY the function implementation inside a "
        f"```python``` code block. Do not include any explanation.\n\n"
        f"Task: {p['prompt']}\n\n"
        f"Example test cases:\n" + "\n".join(p["test_list"][:2])
        for p in problems
    ]

    start = time.time()
    gen_results = engine.generate(prompts, max_tokens=max_tokens)
    gen_time = time.time() - start

    result = BenchmarkResult(
        benchmark="mbpp", config_name=config_name,
        model_path=engine.config.model_path,
        total_problems=len(problems), total_time=gen_time,
    )

    for problem, gen in tqdm(zip(problems, gen_results), total=len(problems), desc="MBPP"):
        raw_code = _extract_code_from_response(gen.output)
        test_code = "\n".join(problem["test_list"])
        passed, error = _execute_code_safely(raw_code, test_code)
        result.results.append(EvalResult(
            task_id=problem["task_id"], prompt=problem["prompt"],
            generated_code=raw_code, passed=passed, error=error,
        ))
        if passed:
            result.passed += 1
        elif error:
            result.errors += 1
        else:
            result.failed += 1

    result.pass_at_1 = result.passed / result.total_problems
    logger.info(f"MBPP pass@1: {result.pass_at_1:.1%} ({result.passed}/{result.total_problems})")
    return result


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def save_result(result: BenchmarkResult, output_dir: str) -> str:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / f"{result.benchmark}_{result.config_name}.json"
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
            {"task_id": r.task_id, "passed": r.passed,
             "generated_code": r.generated_code, "error": r.error}
            for r in result.results
        ],
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    return str(filepath)


def load_result(filepath: str) -> BenchmarkResult:
    with open(filepath) as f:
        data = json.load(f)
    result = BenchmarkResult(
        benchmark=data["benchmark"], config_name=data["config_name"],
        model_path=data["model_path"], total_problems=data["total_problems"],
        passed=data["passed"], failed=data["failed"], errors=data["errors"],
        pass_at_1=data["pass_at_1"],
        exact_match_rate=data.get("exact_match_rate"),
        total_time=data["total_time"],
    )
    for r in data["results"]:
        result.results.append(EvalResult(
            task_id=r["task_id"], prompt="",
            generated_code=r["generated_code"],
            passed=r["passed"], error=r.get("error"),
        ))
    return result


def compute_exact_match(baseline: list[EvalResult], other: list[EvalResult]) -> float:
    assert len(baseline) == len(other)
    matches = sum(1 for b, q in zip(baseline, other) if b.generated_code.strip() == q.generated_code.strip())
    return matches / len(baseline)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results_dir: str = "./results", output_file: str = "./results/binary_summary.md"):
    """
    Generate comparison report for binary configs, including reference
    results from baseline/uniform-int4/boundary-v3 if available.
    """
    results_path = Path(results_dir)

    # Collect all results
    all_results: dict[str, dict[str, BenchmarkResult]] = {}
    for filepath in sorted(results_path.glob("*.json")):
        if filepath.name.startswith("metrics_") or filepath.name.startswith("summary"):
            continue
        try:
            r = load_result(str(filepath))
            if r.config_name not in all_results:
                all_results[r.config_name] = {}
            all_results[r.config_name][r.benchmark] = r
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")

    # Filter to binary + reference configs
    show_configs = []
    for c in REFERENCE_CONFIGS + BINARY_CONFIGS:
        if c in all_results:
            show_configs.append(c)

    if not show_configs:
        print("No results found.")
        return

    # Estimated bits/param for each config
    from quantize_binary import _est_avg_bits, CONFIGS as BIN_CONFIGS
    bits_map = {
        "baseline": 16.0,
        "uniform-int4": 4.0,
        "boundary-v3": 4.50,
    }
    for c in BIN_CONFIGS:
        bits_map[c] = _est_avg_bits(c)

    # Build table
    lines = [
        "# Binary + INT4 Mixed Quantization Results",
        "",
        "FP16 boundary layers (0, 1, 62, 63). Middle layers mixed INT4 + 1-bit.",
        "",
        "## Summary",
        "",
    ]

    # Header
    header = "| Config | Avg bits | HumanEval pass@1 | MBPP pass@1 | HE exact match | MBPP exact match |"
    separator = "|--------|----------|------------------|-------------|-----------------|------------------|"
    lines.extend([header, separator])

    baseline_he = all_results.get("baseline", {}).get("humaneval")
    baseline_mbpp = all_results.get("baseline", {}).get("mbpp")

    for config_name in show_configs:
        benchmarks = all_results[config_name]
        bits = bits_map.get(config_name, "?")
        bits_str = f"{bits:.1f}" if isinstance(bits, float) else bits

        he = benchmarks.get("humaneval")
        mbpp = benchmarks.get("mbpp")

        he_str = f"{he.pass_at_1:.1%} ({he.passed}/{he.total_problems})" if he else "—"
        mbpp_str = f"{mbpp.pass_at_1:.1%} ({mbpp.passed}/{mbpp.total_problems})" if mbpp else "—"

        # Exact match vs baseline
        he_em = "—"
        mbpp_em = "—"
        if config_name != "baseline":
            if baseline_he and he:
                he_em = f"{compute_exact_match(baseline_he.results, he.results):.1%}"
            if baseline_mbpp and mbpp:
                mbpp_em = f"{compute_exact_match(baseline_mbpp.results, mbpp.results):.1%}"

        lines.append(f"| {config_name} | {bits_str} | {he_str} | {mbpp_str} | {he_em} | {mbpp_em} |")

    lines.append("")

    # Layer map for each binary config
    from quantize_binary import print_layer_map
    lines.append("## Layer Maps")
    lines.append("")
    for c in BINARY_CONFIGS:
        lines.append(f"```")
        lines.append(print_layer_map(c))
        lines.append(f"```")
        lines.append("")

    # Regressions vs baseline
    if baseline_he:
        baseline_passed = {r.task_id for r in baseline_he.results if r.passed}
        lines.append("## HumanEval Regressions vs Baseline")
        lines.append("")
        by_task: dict[str, list[str]] = {}
        for config_name in BINARY_CONFIGS:
            he = all_results.get(config_name, {}).get("humaneval")
            if not he:
                continue
            for r in he.results:
                if r.task_id in baseline_passed and not r.passed:
                    by_task.setdefault(r.task_id, []).append(config_name)
        if by_task:
            for tid, configs in sorted(by_task.items()):
                lines.append(f"- **{tid}**: fails in {', '.join(configs)}")
        else:
            lines.append("No regressions found.")
        lines.append("")

    report = "\n".join(lines)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(report)
    logger.info(f"Report saved to {output_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate binary + INT4 mixed quantization models"
    )
    parser.add_argument(
        "--configs", nargs="+", default=BINARY_CONFIGS,
        choices=BINARY_CONFIGS + ["all"],
        help="Which binary configs to evaluate",
    )
    parser.add_argument("--model-dir", default="./models/quantized",
                        help="Base dir containing quantized model folders")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument(
        "--benchmarks", nargs="+", default=["humaneval", "mbpp"],
        choices=["humaneval", "mbpp"],
    )
    parser.add_argument("--report-only", action="store_true",
                        help="Skip evaluation, just regenerate report")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.report_only:
        generate_report(args.output_dir)
        return

    configs = BINARY_CONFIGS if "all" in args.configs else args.configs

    for config_name in configs:
        model_path = os.path.join(args.model_dir, config_name)
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}. Run quantize_binary.py first.")
            continue

        logger.info(f"=== Evaluating {config_name} ===")

        inf_config = InferenceConfig(
            model_path=model_path,
            tensor_parallel_size=args.tp,
        )
        engine = VLLMInference(inf_config)
        engine.load()

        if "humaneval" in args.benchmarks:
            he = eval_humaneval(engine, config_name, os.path.join(args.data_dir, "humaneval"))
            save_result(he, args.output_dir)
            print(f"  HumanEval: {he.pass_at_1:.1%} ({he.passed}/{he.total_problems})")

        if "mbpp" in args.benchmarks:
            mbpp = eval_mbpp(engine, config_name, os.path.join(args.data_dir, "mbpp"))
            save_result(mbpp, args.output_dir)
            print(f"  MBPP: {mbpp.pass_at_1:.1%} ({mbpp.passed}/{mbpp.total_problems})")

        engine.unload()
        print()

    # Generate comparison report
    generate_report(args.output_dir)


if __name__ == "__main__":
    main()
