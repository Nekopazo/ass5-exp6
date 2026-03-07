#!/usr/bin/env python3
"""Run style-encoder pretraining sweep over reference count with fixed 3-token encoder."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def parse_int_csv(raw: str) -> List[int]:
    out: List[int] = []
    for p in str(raw).split(","):
        t = p.strip()
        if not t:
            continue
        out.append(int(t))
    if not out:
        raise ValueError(f"empty integer list: {raw!r}")
    return out


def parse_metrics(metrics_path: Path) -> Dict[str, Any]:
    best_val = None
    best_monitor = None
    best_retr = None
    best_step = 0
    best_step_monitor = 0
    best_step_retr = 0
    last_step = 0
    if not metrics_path.exists():
        return {
            "best_val_loss": None,
            "best_val_monitor": None,
            "best_val_retr_top1": None,
            "best_step": 0,
            "best_step_monitor": 0,
            "best_step_retr": 0,
            "last_step": 0,
        }
    with metrics_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            step = int(row.get("step", 0))
            last_step = max(last_step, step)
            val = row.get("val_loss", None)
            if val is None:
                continue
            val_f = float(val)
            if best_val is None or val_f < best_val:
                best_val = val_f
                best_step = step
            vm = row.get("val_monitor", None)
            if vm is not None:
                vm_f = float(vm)
                if best_monitor is None or vm_f > best_monitor:
                    best_monitor = vm_f
                    best_step_monitor = step
            vr = row.get("val_retr_top1", None)
            if vr is not None:
                vr_f = float(vr)
                if best_retr is None or vr_f > best_retr:
                    best_retr = vr_f
                    best_step_retr = step
    return {
        "best_val_loss": best_val,
        "best_val_monitor": best_monitor,
        "best_val_retr_top1": best_retr,
        "best_step": best_step,
        "best_step_monitor": best_step_monitor,
        "best_step_retr": best_step_retr,
        "last_step": last_step,
    }


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys = [
        "run_name",
        "ref_per_style",
        "style_token_count",
        "return_code",
        "best_val_loss",
        "best_val_monitor",
        "best_val_retr_top1",
        "best_step",
        "best_step_monitor",
        "best_step_retr",
        "last_step",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--pretrain-script", type=Path, default=Path("scripts/pretrain_style_encoder.py"))
    parser.add_argument("--save-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--refs", type=str, default="8,12")
    parser.add_argument("--tokens", type=str, default="3")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--loader-steps-per-epoch", type=int, default=2048)
    parser.add_argument("--val-loader-steps-per-epoch", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--train-font-count", type=int, default=0)
    parser.add_argument("--val-font-count", type=int, default=0)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--rank-metric", type=str, default="best_val_loss", choices=["best_val_monitor", "best_val_retr_top1", "best_val_loss"])
    parser.add_argument("--decode-cache-size", type=int, default=3000)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--worker-torch-threads", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = args.project_root.resolve()
    pretrain_script = args.pretrain_script
    if not pretrain_script.is_absolute():
        pretrain_script = (root / pretrain_script).resolve()
    if not pretrain_script.exists():
        raise FileNotFoundError(f"pretrain script not found: {pretrain_script}")

    refs = parse_int_csv(args.refs)
    tokens = parse_int_csv(args.tokens)
    if any(int(tok) != 3 for tok in tokens):
        raise ValueError("style-token-count is now fixed to 3 (t_low/t_mid/t_high)")

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sweep_name = args.run_name.strip() or f"style_pretrain_sweep_{stamp}"
    save_root = args.save_root
    if not save_root.is_absolute():
        save_root = (root / save_root).resolve()
    sweep_dir = (save_root / sweep_name).resolve()
    sweep_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for ref in refs:
        for tok in tokens:
            run_name = f"ref{int(ref)}_t{int(tok)}"
            run_dir = (sweep_dir / run_name).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)

            out_ckpt = run_dir / "final.pt"
            log_file = run_dir / "train.log"
            metrics_jsonl = run_dir / "metrics.jsonl"

            cmd = [
                sys.executable,
                str(pretrain_script),
                "--project-root",
                str(root),
                "--out",
                str(out_ckpt),
                "--log-file",
                str(log_file),
                "--metrics-jsonl",
                str(metrics_jsonl),
                "--steps",
                str(int(args.steps)),
                "--batch-size",
                str(int(args.batch_size)),
                "--loader-steps-per-epoch",
                str(int(args.loader_steps_per_epoch)),
                "--val-loader-steps-per-epoch",
                str(int(args.val_loader_steps_per_epoch)),
                "--val-ratio",
                str(float(args.val_ratio)),
                "--train-font-count",
                str(int(args.train_font_count)),
                "--val-font-count",
                str(int(args.val_font_count)),
                "--val-batches",
                str(int(args.val_batches)),
                "--log-every",
                str(int(args.log_every)),
                "--decode-cache-size",
                str(int(args.decode_cache_size)),
                "--decode-workers",
                str(int(args.decode_workers)),
                "--prefetch-factor",
                str(int(args.prefetch_factor)),
                "--worker-torch-threads",
                str(int(args.worker_torch_threads)),
                "--ref-per-style",
                str(int(ref)),
                "--style-token-count",
                str(int(tok)),
                "--device",
                str(args.device),
                "--seed",
                str(int(args.seed)),
            ]

            print(
                f"[sweep] start {run_name} "
                f"(ref={int(ref)} token={int(tok)})",
                flush=True,
            )
            rc = subprocess.run(cmd, cwd=str(root), check=False).returncode

            metric_stat = parse_metrics(metrics_jsonl)

            row: Dict[str, Any] = {
                "run_name": run_name,
                "ref_per_style": int(ref),
                "style_token_count": int(tok),
                "return_code": int(rc),
                "best_val_loss": metric_stat["best_val_loss"],
                "best_val_monitor": metric_stat["best_val_monitor"],
                "best_val_retr_top1": metric_stat["best_val_retr_top1"],
                "best_step": int(metric_stat["best_step"]),
                "best_step_monitor": int(metric_stat["best_step_monitor"]),
                "best_step_retr": int(metric_stat["best_step_retr"]),
                "last_step": int(metric_stat["last_step"]),
                "run_dir": str(run_dir),
            }
            rows.append(row)

            print(
                f"[sweep] done {run_name} rc={rc} "
                f"best_val={row['best_val_loss']} best_mon={row['best_val_monitor']} "
                f"best_retr={row['best_val_retr_top1']}",
                flush=True,
            )

    rank_key = str(args.rank_metric)
    ranking = [r for r in rows if r["return_code"] == 0 and r.get(rank_key) is not None]
    reverse = rank_key != "best_val_loss"
    ranking.sort(key=lambda x: float(x[rank_key]), reverse=reverse)

    summary_json = sweep_dir / "summary.json"
    summary_csv = sweep_dir / "summary.csv"
    payload = {
        "sweep_dir": str(sweep_dir),
        "refs": refs,
        "tokens": tokens,
        "steps": int(args.steps),
        "loader_steps_per_epoch": int(args.loader_steps_per_epoch),
        "val_loader_steps_per_epoch": int(args.val_loader_steps_per_epoch),
        "rank_metric": str(rank_key),
        "rows": rows,
        "ranking": ranking,
        "best": ranking[0] if ranking else None,
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(summary_csv, rows)

    print(f"[sweep] summary_json={summary_json}", flush=True)
    print(f"[sweep] summary_csv={summary_csv}", flush=True)
    if ranking:
        best = ranking[0]
        best_overall_ckpt = sweep_dir / "best_overall.pt"
        best_overall_meta = sweep_dir / "best_overall.json"
        src_final = Path(str(best["run_dir"])) / "final.pt"
        ckpt_source = "none"
        if src_final.exists():
            shutil.copy2(src_final, best_overall_ckpt)
            ckpt_source = "final.pt"
        best_overall_meta.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            "[sweep] best "
            f"run={best['run_name']} ref={best['ref_per_style']} "
            f"token={best['style_token_count']} {rank_key}={best[rank_key]}",
            flush=True,
        )
        print(f"[sweep] best_overall_ckpt={best_overall_ckpt} source={ckpt_source}", flush=True)
        print(f"[sweep] best_overall_meta={best_overall_meta}", flush=True)
    else:
        print(f"[sweep] no successful runs with {rank_key}", flush=True)


if __name__ == "__main__":
    main()
