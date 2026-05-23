#!/usr/bin/env python3
"""Summarize module-level gradient and Adam diagnostics from training logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def mean_metric(rows: list[dict], module_name: str, metric_name: str) -> float:
    values: list[float] = []
    for row in rows:
        value = row.get("modules", {}).get(module_name, {}).get(metric_name)
        if value is not None:
            values.append(float(value))
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_file", type=Path)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    rows = load_rows(args.stats_file)
    if not rows:
        raise SystemExit(f"no rows found in {args.stats_file}")
    window = max(1, min(int(args.window), len(rows)))
    top_k = max(1, int(args.top_k))
    tail_rows = rows[-window:]
    module_names = sorted({name for row in tail_rows for name in row.get("modules", {})})

    metrics = [
        "grad_to_param_ratio",
        "grad_norm",
        "grad_abs_max",
        "adam_update_rms",
        "adam_exp_avg_sq_rms",
    ]

    print(f"file: {args.stats_file}")
    print(f"rows: {len(rows)} window: {window} last_step: {rows[-1]['step']}")
    for metric_name in metrics:
        ranked = sorted(
            ((module_name, mean_metric(tail_rows, module_name, metric_name)) for module_name in module_names),
            key=lambda item: item[1],
            reverse=True,
        )
        print(f"\n[{metric_name}]")
        for module_name, value in ranked[:top_k]:
            print(f"{module_name}\t{value:.8e}")


if __name__ == "__main__":
    main()
