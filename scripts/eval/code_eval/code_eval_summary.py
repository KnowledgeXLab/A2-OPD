#!/usr/bin/env python3
"""Build or update code eval summary.json."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from datetime import datetime
from pathlib import Path

LABELS = {
    "humaneval": "HumanEval+",
    "mbpp": "MBPP+",
    "lcb": "LCB v6",
}


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def load_lcb_eval_all_results(eval_all_path: Path) -> list | None:
    for attempt in range(10):
        try:
            if not eval_all_path.is_file():
                raise FileNotFoundError(eval_all_path)
            results = json.loads(eval_all_path.read_text(encoding="utf-8"))
            break
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            if attempt == 9:
                raise
            time.sleep(1)
    if not isinstance(results, list) or not results:
        return None
    return results


def lcb_metrics_from_results(results: list) -> tuple[float | None, int]:
    scores = []
    for item in results:
        graded = item.get("graded_list") or []
        if not graded:
            continue
        scores.append(estimate_pass_at_k(len(graded), sum(bool(g) for g in graded), 1))
    if not scores:
        return None, len(results)
    return sum(scores) / len(scores), len(results)


def lcb_pass_at_1(eval_all_path: Path) -> float | None:
    results = load_lcb_eval_all_results(eval_all_path)
    if results is None:
        return None
    p1, _ = lcb_metrics_from_results(results)
    return p1


def find_lcb_eval_all(
    output_root: Path,
    n_samples: int,
    temperature: float,
    lcb_source_dir: Path | None = None,
) -> Path | None:
    patterns = [
        f"Scenario.codegeneration_{n_samples}_{temperature}_eval_all.json",
        f"codegeneration_{n_samples}_{temperature}_eval_all.json",
    ]
    for lcb_dir in (output_root / "lcb", lcb_source_dir):
        if lcb_dir is None or not Path(lcb_dir).is_dir():
            continue
        lcb_dir = Path(lcb_dir)
        for pattern in patterns:
            candidate = lcb_dir / pattern
            if candidate.is_file():
                return candidate
    return None


def load_evalplus_metrics(result_file: Path) -> dict | None:
    if not result_file.is_file():
        return None
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    plus = data.get("pass_at_k", {}).get("plus", {})
    base = data.get("pass_at_k", {}).get("base", {})
    p1 = plus.get("pass@1")
    if p1 is None:
        p1 = base.get("pass@1")
    if p1 is None:
        return None
    num_problems = len(data.get("eval", {}))
    return {
        "pass@1": p1,
        "pass@1_pct": p1 * 100.0,
        "num_problems": num_problems,
    }


def load_lcb_metrics(
    output_root: Path,
    n_samples: int,
    temperature: float,
    lcb_source_dir: Path | None,
) -> dict | None:
    eval_all = find_lcb_eval_all(output_root, n_samples, temperature, lcb_source_dir)
    if eval_all is None:
        return None
    try:
        results = load_lcb_eval_all_results(eval_all)
    except Exception:
        return None
    if results is None:
        return None
    p1, num_problems = lcb_metrics_from_results(results)
    if p1 is None:
        return None
    return {
        "pass@1": p1,
        "pass@1_pct": p1 * 100.0,
        "num_problems": num_problems,
        "eval_all": str(eval_all),
    }


def _bench_num_problems(metric: dict | None) -> int:
    if metric is None:
        return 0
    return int(metric.get("num_problems") or 0)


def recompute_averages(metrics: dict, verbose: bool) -> None:
    evalplus_rows = [
        metrics[key]["pass@1_pct"]
        for key in ("humaneval", "mbpp")
        if key in metrics and metrics[key].get("pass@1_pct") is not None
    ]
    if evalplus_rows:
        metrics["evalplus_avg_pass@1_pct"] = sum(evalplus_rows) / len(evalplus_rows)
        if verbose:
            avg = metrics["evalplus_avg_pass@1_pct"]
            n_benches = len(evalplus_rows)
            total_problems = sum(
                _bench_num_problems(metrics.get(key))
                for key in ("humaneval", "mbpp")
                if key in metrics and metrics[key].get("pass@1_pct") is not None
            )
            print(
                f"EvalPlus Avg (pass@1, {n_benches} bench, {total_problems} problems): "
                f"{avg:.2f}%"
            )
    elif "evalplus_avg_pass@1_pct" in metrics:
        del metrics["evalplus_avg_pass@1_pct"]

    code_avg_parts = [
        metrics[key]["pass@1_pct"]
        for key in ("humaneval", "mbpp", "lcb")
        if key in metrics and metrics[key].get("pass@1_pct") is not None
    ]
    if len(code_avg_parts) == 3:
        metrics["code_avg_pass@1_pct"] = sum(code_avg_parts) / 3
        if verbose:
            total_problems = sum(
                _bench_num_problems(metrics.get(key))
                for key in ("humaneval", "mbpp", "lcb")
                if key in metrics and metrics[key].get("pass@1_pct") is not None
            )
            print(
                f"Code Avg (HE+ / MBPP+ / LCB, {total_problems} problems): "
                f"{metrics['code_avg_pass@1_pct']:.2f}%"
            )
    elif "code_avg_pass@1_pct" in metrics:
        del metrics["code_avg_pass@1_pct"]


def update_summary(
    output_root: str | Path,
    model_name: str,
    enabled_benches: list[str],
    *,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    batch_size: int,
    lcb_source_dir: str | Path | None = None,
    verbose: bool = True,
) -> Path:
    output_root = Path(output_root)
    lcb_source = Path(lcb_source_dir) if lcb_source_dir else None
    summary_path = output_root / "summary.json"
    lock_path = summary_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            prev: dict = {}
            if summary_path.is_file():
                try:
                    prev = json.loads(summary_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    prev = {}

            summary = {
                "model": model_name,
                "n_samples": n_samples,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "batch_size": batch_size,
                "output_root": str(output_root),
                "date": os.environ.get("OPD_LOG_DATETIME", datetime.now().strftime("%Y-%m-%d %H:%M")),
                "benches_run": dict(prev.get("benches_run", {})),
                "metrics": dict(prev.get("metrics", {})),
            }

            if verbose:
                print("\n=== Code Eval Summary ===")
                print(
                    f"model={model_name}  (n={n_samples}, temp={temperature}, "
                    f"top_p={top_p}, max_tokens={max_tokens}, batch_size={batch_size})"
                )
                print(f"output_root={output_root}")

            for bench in enabled_benches:
                summary["benches_run"][bench] = True
                label = LABELS.get(bench, bench)

                if bench in ("humaneval", "mbpp"):
                    result_file = output_root / bench / "eval_results.json"
                    metric = load_evalplus_metrics(result_file)
                    if metric is None:
                        if verbose and result_file.is_file():
                            print(
                                f"{label}: pass@1 not found in {result_file}  (n=0)"
                            )
                        elif verbose:
                            print(f"{label}: missing {result_file}  (n=0)")
                        continue
                    summary["metrics"][bench] = metric
                    if verbose:
                        print(
                            f"{label}: pass@1 = {metric['pass@1_pct']:.2f}%  "
                            f"(n={metric['num_problems']})"
                        )
                    continue

                if bench == "lcb":
                    metric = load_lcb_metrics(output_root, n_samples, temperature, lcb_source)
                    if metric is None:
                        if verbose:
                            print(
                                f"{label}: missing eval_all under {output_root / 'lcb'}  "
                                f"(n=0)"
                            )
                        continue
                    summary["metrics"]["lcb"] = metric
                    if verbose:
                        eval_name = Path(metric["eval_all"]).name
                        print(
                            f"{label}: pass@1 = {metric['pass@1_pct']:.2f}%  "
                            f"(n={metric['num_problems']}, {eval_name})"
                        )

            recompute_averages(summary["metrics"], verbose=verbose)
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
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
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lcb_source_dir", default="")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    benches = [x for x in args.benches.split(",") if x]
    lcb_source_dir = args.lcb_source_dir or None
    update_summary(
        args.output_root,
        args.model_name,
        benches,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        lcb_source_dir=lcb_source_dir,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
