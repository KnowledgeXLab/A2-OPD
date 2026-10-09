#!/usr/bin/env python3
"""Resample n rollout results to k (< n) and recompute math eval metrics."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from math_eval_utils import load_saved_results_by_key


def subsample_row(row: dict, k: int, rng: random.Random) -> dict:
    accs = row.get("acc_list") or []
    responses = row.get("responses") or []
    preds = row.get("pred_answers") or []

    n = len(accs)
    if n < k:
        raise ValueError(f"sample id={row.get('id')} has only {n} rollouts, need k={k}")

    idxs = sorted(rng.sample(range(n), k))
    new_row = dict(row)
    new_row["acc_list"] = [accs[i] for i in idxs]
    if responses:
        new_row["responses"] = [responses[i] for i in idxs]
    if preds:
        new_row["pred_answers"] = [preds[i] for i in idxs]
    new_row["resample_indices"] = idxs
    return new_row


def compute_metrics(rows: list[dict]) -> dict:
    total = 0
    correct = 0
    pass_any = 0
    count = 0

    for row in rows:
        accs = row.get("acc_list") or []
        if not accs:
            continue
        total += len(accs)
        correct += sum(1 for x in accs if x)
        if any(accs):
            pass_any += 1
        count += 1

    if total == 0:
        return {}

    return {
        "num_samples": count,
        "accuracy": correct / total,
        "accuracy_pct": correct / total * 100.0,
        "pass@k": pass_any / count if count else 0.0,
        "pass@k_pct": pass_any / count * 100.0 if count else 0.0,
    }


def find_bench_jsonls(output_root: Path) -> list[tuple[str, Path]]:
    benches: list[tuple[str, Path]] = []
    for bench_dir in sorted(output_root.iterdir()):
        if not bench_dir.is_dir() or bench_dir.name.startswith("."):
            continue
        jsonls = sorted(bench_dir.glob("*.jsonl"))
        if len(jsonls) == 1:
            benches.append((bench_dir.name, jsonls[0]))
        elif len(jsonls) > 1:
            benches.append((bench_dir.name, jsonls[0]))
    return benches


def resample_bench(
    result_file: Path,
    k: int,
    source_n: int,
    rng: random.Random,
) -> tuple[list[dict], dict]:
    saved_by_key, _ = load_saved_results_by_key(result_file)
    rows = list(saved_by_key.values())
    if not rows:
        return [], {}

    resampled: list[dict] = []
    for row in rows:
        accs = row.get("acc_list") or []
        if len(accs) != source_n:
            raise ValueError(
                f"{result_file}: id={row.get('id')} has {len(accs)} rollouts, "
                f"expected source_n={source_n}"
            )
        resampled.append(subsample_row(row, k, rng))

    return resampled, compute_metrics(resampled)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Randomly subsample k rollouts from n=32 results and recompute metrics."
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Model eval output dir, e.g. .../math_eval_outputs/Qwen3-4B-...",
    )
    parser.add_argument("--k", type=int, default=16, help="Subsample size (default: 16)")
    parser.add_argument(
        "--source_n",
        type=int,
        default=32,
        help="Original rollout count per problem (default: 32)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save_json",
        default="",
        help="Optional output summary json path; default: <output_root>/summary_pass{k}_n{source_n}.json",
    )
    parser.add_argument(
        "--save_resampled_dir",
        default="",
        help="Optional dir to save resampled jsonl per bench",
    )
    args = parser.parse_args()

    if args.k <= 0 or args.k > args.source_n:
        raise SystemExit(f"Require 0 < k <= source_n, got k={args.k}, source_n={args.source_n}")

    output_root = Path(args.output_root)
    if not output_root.is_dir():
        raise SystemExit(f"Not a directory: {output_root}")

    benches = find_bench_jsonls(output_root)
    if not benches:
        raise SystemExit(f"No bench jsonl found under {output_root}")

    rng = random.Random(args.seed)
    summary = {
        "output_root": str(output_root),
        "source_n": args.source_n,
        "k": args.k,
        "seed": args.seed,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "metrics": {},
    }

    print(f"output_root={output_root}")
    print(f"resample: n={args.source_n} -> k={args.k}, seed={args.seed}")
    print()

    acc_values = []
    for bench, result_file in benches:
        try:
            resampled_rows, metric = resample_bench(
                result_file, args.k, args.source_n, rng
            )
        except ValueError as exc:
            print(f"[{bench}] skip: {exc}")
            continue

        if not metric:
            print(f"[{bench}] no valid samples in {result_file}")
            continue

        summary["metrics"][bench] = metric
        acc_values.append(metric["accuracy_pct"])
        print(
            f"[{bench}] accuracy={metric['accuracy_pct']:.2f}%  "
            f"pass@{args.k}={metric['pass@k_pct']:.2f}%  "
            f"(n={metric['num_samples']})"
        )

        if args.save_resampled_dir:
            out_dir = Path(args.save_resampled_dir) / bench
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / result_file.name
            with out_path.open("w", encoding="utf-8") as f:
                for row in resampled_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if acc_values:
        avg = sum(acc_values) / len(acc_values)
        summary["metrics"]["math_avg_accuracy_pct"] = avg
        print(f"\nMath Avg (accuracy): {avg:.2f}%")

    save_path = Path(args.save_json) if args.save_json else (
        output_root / f"summary_pass{args.k}_n{args.source_n}.json"
    )
    save_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nsummary saved to {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
