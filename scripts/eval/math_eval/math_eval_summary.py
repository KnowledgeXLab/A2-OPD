#!/usr/bin/env python3
"""Build or update math eval summary.json."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from datetime import datetime
from pathlib import Path

from math_eval_utils import bench_output_dir, load_saved_results_by_key, result_file_for

LABELS = {
    "AIME2024": "AIME2024",
    "AIME2025": "AIME2025",
    "AIME2026": "AIME2026",
    "SMT2025": "SMT2025",
    "CMIMC2025": "CMIMC2025",
    "HMMT2025FEB": "HMMT2025(Feb)",
    "HMMT2025NOV": "HMMT2025(Nov)",
    "HMMT2026FEB": "HMMT2026(Feb)",
}


def load_metrics(result_file: Path):
    saved_by_key, _ = load_saved_results_by_key(result_file)
    rows = list(saved_by_key.values())
    if not rows:
        return None

    total = 0
    correct = 0
    pass_any = 0
    count = 0
    for row in rows:
        if not (row.get("acc_list") and row.get("responses")):
            continue
        accs = row.get("acc_list") or []
        total += len(accs)
        correct += sum(1 for x in accs if x)
        if any(accs):
            pass_any += 1
        count += 1
    if total == 0:
        return None
    return {
        "num_samples": count,
        "accuracy": correct / total,
        "accuracy_pct": correct / total * 100.0,
        "pass@k": pass_any / count if count else 0.0,
        "pass@k_pct": pass_any / count * 100.0 if count else 0.0,
    }


def update_summary(
    output_root: str | Path,
    model_name: str,
    enabled_benches: list[str],
    *,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    verbose: bool = True,
) -> Path:
    output_root = Path(output_root)
    summary_path = output_root / "summary.json"
    lock_path = summary_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            prev: dict = {}
            if summary_path.is_file():
                try:
                    prev = json.loads(summary_path.read_text())
                except json.JSONDecodeError:
                    prev = {}

            summary = {
                "model": model_name,
                "n_samples": n_samples,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "output_root": str(output_root),
                "date": os.environ.get("OPD_LOG_DATETIME", datetime.now().strftime("%Y-%m-%d %H:%M")),
                "benches_run": dict(prev.get("benches_run", {})),
                "metrics": dict(prev.get("metrics", {})),
            }

            if verbose:
                print("\n=== Math Eval Summary ===")
                print(
                    f"model={model_name}  (n={n_samples}, temp={temperature}, "
                    f"top_p={top_p}, max_tokens={max_tokens})"
                )
                print(f"output_root={output_root}")

            for dataset in enabled_benches:
                summary["benches_run"][bench_output_dir(dataset)] = True
                label = LABELS.get(dataset, dataset)
                result_file = result_file_for(output_root, dataset, model_name)
                if not result_file.is_file():
                    if verbose:
                        print(f"{label}: missing {result_file}")
                    continue
                metric = load_metrics(result_file)
                if metric is None:
                    if verbose:
                        print(f"{label}: no predictions in {result_file}")
                    continue
                summary["metrics"][bench_output_dir(dataset)] = metric
                if verbose:
                    print(
                        f"{label}: accuracy = {metric['accuracy_pct']:.2f}%  "
                        f"pass@k = {metric['pass@k_pct']:.2f}%  (n={metric['num_samples']})"
                    )

            acc_pcts = [
                summary["metrics"][bench_output_dir(dataset)]["accuracy_pct"]
                for dataset in enabled_benches
                if bench_output_dir(dataset) in summary["metrics"]
            ]
            if acc_pcts:
                summary["metrics"]["math_avg_accuracy_pct"] = sum(acc_pcts) / len(acc_pcts)
                if verbose:
                    avg = summary["metrics"]["math_avg_accuracy_pct"]
                    print(f"Math Avg (accuracy): {avg:.2f}%")

            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    if verbose:
        print(f"summary saved to {summary_path}")
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--benches", required=True, help="Comma-separated bench names")
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top_p", type=float, required=True)
    parser.add_argument("--max_tokens", type=int, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    benches = [x for x in args.benches.split(",") if x]
    update_summary(
        args.output_root,
        args.model_name,
        benches,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
