#!/usr/bin/env python3
"""Overnight-decision driver: extended micro-pretraining + deep headroom diagnostics
+ the v4 oracle gating ladder.

Sequences:
  1. micro_pretrain_eval.py  (--methods --epochs --cutoff)  -> deep headroom per method
  2. oracle_gating_eval.py   (v4 ladder, 1M-point pool)     -> per trained method

Outputs land in logs/ (train log, one ladder log per method) and
<out_dir>/micro_pretrain_results.json + <out_dir>/<method>/oracle_gating_results.json

Usage (GPU server, from repo root):
  uv run python long_diagnostics_eval.py --methods supcon_vib,supcon_vib_strongvib,supcon_vib_additive --epochs 30
"""
import argparse
import os
import subprocess
import sys
import time

def run_step(cmd, log_path):
    print(f"\n{'='*72}\n$ {' '.join(cmd)}\n  -> {log_path}\n{'='*72}", flush=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print(f"STEP FAILED (rc={p.returncode}): {' '.join(cmd)}\n  see {log_path}", flush=True)
        sys.exit(1)
    print(f"done: {log_path}", flush=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", default="supcon_vib,supcon_vib_strongvib,supcon_vib_additive")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--cutoff", type=float, default=0.1)
    ap.add_argument("--out_dir", default="logs/micro_pretrain_long")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_ladder", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    print(f"Methods: {methods} | epochs: {args.epochs} | cutoff: {args.cutoff}")

    if not args.skip_train:
        run_step(["uv", "run", "python", "micro_pretrain_eval.py",
                  f"--methods={','.join(methods)}",
                  f"--epochs={args.epochs}",
                  f"--cutoff={args.cutoff}",
                  f"--out_dir={args.out_dir}"],
                 "logs/long_diagnostics_train.log")

    if not args.skip_ladder:
        for m in methods:
            load = os.path.join(args.out_dir, m)
            if not os.path.exists(load):
                print(f"skip ladder for {m}: {load} not found", flush=True)
                continue
            run_step(["uv", "run", "python", "oracle_gating_eval.py",
                      f"--load_path={load}",
                      f"--method={m}"],
                     f"logs/oracle_gating_{m}_long.log")

    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
