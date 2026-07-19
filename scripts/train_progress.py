#!/usr/bin/env python3
"""Print progress + ETA for a perception training run (Phase 2).

Reads Ultralytics' ``results.csv`` (one row per finished epoch) from the most recent run under
``outputs/perception/`` — or a run dir passed as the first argument — and reports epochs done,
average time per epoch, elapsed, ETA, and the latest mAP. Pure stdlib; run it anytime while
training, in a second terminal:

    python3 scripts/train_progress.py
    make progress
"""

from __future__ import annotations

import csv
import glob
import os
import sys


def find_latest_run() -> str | None:
    runs = sorted(glob.glob("outputs/perception/*/"), key=os.path.getmtime)
    return runs[-1] if runs else None


def total_epochs(run: str) -> int | None:
    args = os.path.join(run, "args.yaml")
    if os.path.exists(args):
        for line in open(args):
            if line.strip().startswith("epochs:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except ValueError:
                    return None
    return None


def fmt(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}h {s % 3600 // 60}m"


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else find_latest_run()
    if not run or not os.path.isdir(run):
        print("No run dir found under outputs/perception/ — has training started?")
        return
    print(f"run: {run}")

    csv_path = os.path.join(run, "results.csv")
    if not os.path.exists(csv_path):
        print("Still inside epoch 1 (no results.csv yet).")
        return
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        print("No epochs logged yet.")
        return

    done = len(rows)
    total = total_epochs(run) or done
    last = {k.strip(): v for k, v in rows[-1].items()}
    elapsed = float(last.get("time", 0.0))
    per = elapsed / done if done else 0.0
    remaining = per * (total - done)

    def metric(*keys: str) -> str:
        for k in keys:
            if k in last and last[k] not in ("", None):
                return last[k]
        return "?"

    pct = 100 * done // total if total else 0
    print(f"epochs   : {done}/{total}  ({pct}%)")
    print(f"per epoch: {per / 60:.1f} min    elapsed: {fmt(elapsed)}    eta: {fmt(remaining)}")
    print(f"mAP@50   : {metric('metrics/mAP50(B)', 'metrics/mAP50(M)')}")
    print(f"mAP@50-95: {metric('metrics/mAP50-95(B)', 'metrics/mAP50-95(M)')}")
    print("(eta assumes all epochs run; early-stopping may finish sooner)")


if __name__ == "__main__":
    main()
