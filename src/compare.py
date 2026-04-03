"""
Comparison and analysis across quantization configs.

Loads results from all configs, computes diffs, and produces
summary tables and per-problem analysis.
"""

import json
import logging
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from evaluate import BenchmarkResult, load_results, compute_exact_match

logger = logging.getLogger(__name__)


def load_all_results(results_dir: str) -> dict[str, dict[str, BenchmarkResult]]:
    """
    Load all benchmark results from a directory.

    Returns nested dict: {config_name: {benchmark_name: BenchmarkResult}}
    """
    results_path = Path(results_dir)
    all_results = {}

    for filepath in sorted(results_path.glob("*.json")):
        if filepath.name.startswith("metrics_"):
            continue  # Skip metrics files
        try:
            result = load_results(str(filepath))
            if result.config_name not in all_results:
                all_results[result.config_name] = {}
            all_results[result.config_name][result.benchmark] = result
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")

    return all_results


def load_all_metrics(results_dir: str) -> dict[str, dict]:
    """Load all metrics files from a directory."""
    results_path = Path(results_dir)
    metrics = {}

    for filepath in sorted(results_path.glob("metrics_*.json")):
        with open(filepath) as f:
            data = json.load(f)
        metrics[data["config_name"]] = data

    return metrics


def build_comparison_table(
    all_results: dict[str, dict[str, BenchmarkResult]],
    all_metrics: dict[str, dict] | None = None,
    baseline_name: str = "baseline",
) -> pd.DataFrame:
    """
    Build a comparison DataFrame across all configs.
    """
    rows = []

    for config_name, benchmarks in sorted(all_results.items()):
        row = {"config": config_name}

        for bench_name in ["humaneval", "mbpp"]:
            if bench_name in benchmarks:
                result = benchmarks[bench_name]
                row[f"{bench_name}_pass@1"] = result.pass_at_1
                row[f"{bench_name}_passed"] = result.passed
                row[f"{bench_name}_total"] = result.total_problems
                row[f"{bench_name}_time_s"] = result.total_time

                # Exact match vs baseline
                if (baseline_name in all_results
                        and bench_name in all_results[baseline_name]
                        and config_name != baseline_name):
                    baseline_results = all_results[baseline_name][bench_name].results
                    em = compute_exact_match(baseline_results, result.results)
                    row[f"{bench_name}_exact_match"] = em

        # Add metrics if available
        if all_metrics and config_name in all_metrics:
            m = all_metrics[config_name]
            row["model_size_gb"] = m.get("model_size_gb")
            row["vram_gb"] = m.get("total_vram_gb")
            row["tok_per_sec"] = m.get("tokens_per_second")
            row["perplexity"] = m.get("perplexity")

        rows.append(row)

    return pd.DataFrame(rows)


def print_summary_table(df: pd.DataFrame) -> str:
    """Pretty-print the comparison table."""
    # Select key columns
    display_cols = ["config"]
    for col in ["humaneval_pass@1", "mbpp_pass@1",
                 "humaneval_exact_match", "mbpp_exact_match",
                 "model_size_gb", "vram_gb", "tok_per_sec", "perplexity"]:
        if col in df.columns:
            display_cols.append(col)

    display_df = df[display_cols].copy()

    # Format percentages
    for col in display_df.columns:
        if "pass@1" in col or "exact_match" in col:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—"
            )

    table = tabulate(display_df, headers="keys", tablefmt="github", showindex=False)
    return table


def analyze_failures(
    all_results: dict[str, dict[str, BenchmarkResult]],
    baseline_name: str = "baseline",
    benchmark: str = "humaneval",
) -> list[dict]:
    """
    Find problems where quantized configs fail but baseline passes.

    Returns list of dicts with problem details and which configs failed.
    """
    if baseline_name not in all_results or benchmark not in all_results[baseline_name]:
        return []

    baseline = all_results[baseline_name][benchmark]
    baseline_passed = {
        r.task_id for r in baseline.results if r.passed
    }

    regressions = []
    for config_name, benchmarks in sorted(all_results.items()):
        if config_name == baseline_name or benchmark not in benchmarks:
            continue

        result = benchmarks[benchmark]
        for r in result.results:
            if r.task_id in baseline_passed and not r.passed:
                regressions.append({
                    "task_id": r.task_id,
                    "config": config_name,
                    "error": r.error,
                    "generated_code_snippet": r.generated_code[:200],
                })

    return regressions


def generate_report(
    results_dir: str = "./results",
    output_file: str = "./results/summary.md",
    baseline_name: str = "baseline",
) -> str:
    """
    Generate a full markdown comparison report.
    """
    all_results = load_all_results(results_dir)
    all_metrics = load_all_metrics(results_dir)

    if not all_results:
        return "No results found."

    df = build_comparison_table(all_results, all_metrics, baseline_name)
    table = print_summary_table(df)

    lines = [
        "# Boundary V Quantization Experiment Results",
        "",
        "## Summary Table",
        "",
        table,
        "",
    ]

    # Failure analysis for each benchmark
    for bench in ["humaneval", "mbpp"]:
        regressions = analyze_failures(all_results, baseline_name, bench)
        if regressions:
            lines.append(f"## {bench.upper()} Regressions vs Baseline")
            lines.append("")
            lines.append(f"Problems where baseline passes but quantized config fails:")
            lines.append("")

            # Group by task_id
            by_task = {}
            for r in regressions:
                tid = r["task_id"]
                if tid not in by_task:
                    by_task[tid] = []
                by_task[tid].append(r["config"])

            for tid, configs in sorted(by_task.items()):
                lines.append(f"- **{tid}**: fails in {', '.join(configs)}")

            lines.append("")

    # Per-config details
    lines.append("## Per-Config Details")
    lines.append("")

    for config_name in sorted(all_results.keys()):
        lines.append(f"### {config_name}")
        lines.append("")

        if config_name in all_metrics:
            m = all_metrics[config_name]
            lines.append(f"- Model size: {m.get('model_size_gb', '?'):.2f} GB")
            lines.append(f"- Peak VRAM: {m.get('total_vram_gb', '?'):.2f} GB")
            lines.append(f"- Throughput: {m.get('tokens_per_second', '?'):.1f} tok/s")
            if m.get("perplexity"):
                lines.append(f"- Perplexity: {m['perplexity']:.3f}")

        for bench_name in ["humaneval", "mbpp"]:
            if bench_name in all_results[config_name]:
                r = all_results[config_name][bench_name]
                lines.append(
                    f"- {bench_name} pass@1: {r.pass_at_1:.1%} "
                    f"({r.passed}/{r.total_problems})"
                )

        lines.append("")

    report = "\n".join(lines)

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    logger.info(f"Report saved to {output_file}")
    print(report)
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--output", default="./results/summary.md")
    parser.add_argument("--baseline", default="baseline")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    generate_report(args.results_dir, args.output, args.baseline)


if __name__ == "__main__":
    main()
