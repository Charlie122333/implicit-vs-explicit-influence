#!/usr/bin/env python3
"""Recreate the paper's detection-rate figures from your results.

Each setting's `plot.py` turns that setting's results (the `.eval` logs — and, for
Settings 2/3, the monitor sidecar — under its `logs/`) into the paper's detection-rate
figure (y-axis = monitor detection rate %, the `*_attribution_rate_v2.pdf` panels):

  Setting 1  -> setting1_binary/figures/S1_attribution_rate.pdf
  Setting 2  -> setting2_rating/figures/S2_rating_attribution_rate.pdf
  Setting 3  -> setting3_mcqa/figures/S3_attribution_rate.pdf
  Setting 4  -> setting4_coding/figures/S4_attribution_rate.pdf
  Load-bear. -> loadbearing/figures/loadbearing_detection.pdf

This driver just runs each `plot.py` with its default conditions. It does NOT generate or
monitor anything — you must have run each setting's benchmark first (see each README), so
that its `logs/` folder holds the results. Settings whose logs are missing are skipped with
a note (this repo ships no result files by design).

Usage:
    python make_figures.py                 # all settings it can find results for
    python make_figures.py --settings 1 4  # a subset (1,2,3,4,lb)
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.resolve()

# key -> (directory, plot script, extra args, human label)
PLOTS = {
    "1":  ("setting1_binary", "plot.py", [], "Setting 1 (binary preference)"),
    "2":  ("setting2_rating", "plot.py", [], "Setting 2 (multi-domain rating)"),
    "3":  ("setting3_mcqa",   "plot.py", [], "Setting 3 (hard MCQA)"),
    "4":  ("setting4_coding", "plot.py", [], "Setting 4 (open-ended coding)"),
    # detection is the paper's main load-bearing figure; the capability figure needs the
    # separate probe (loadbearing/capability_probe.py) — make it with `plot.py --which capability`.
    "lb": ("loadbearing",     "plot.py", ["--which", "detection"], "Load-bearing variant"),
}


def main():
    ap = argparse.ArgumentParser(description="Recreate the paper's detection-rate figures")
    ap.add_argument("--settings", nargs="+", default=list(PLOTS.keys()),
                    choices=list(PLOTS.keys()), help="which figures to make (default: all)")
    args = ap.parse_args()

    made, skipped = [], []
    for key in args.settings:
        subdir, script, extra, label = PLOTS[key]
        path = REPO / subdir / script
        print(f"\n=== {label}: {subdir}/{script} ===")
        proc = subprocess.run([sys.executable, str(path), *extra], cwd=str(REPO / subdir),
                              capture_output=True, text=True)
        out = (proc.stdout + proc.stderr).strip()
        if out:
            print(out)
        # A missing-logs FileNotFoundError is the expected "no results yet" case.
        if proc.returncode == 0:
            made.append(label)
        elif "FileNotFoundError" in out or "Missing" in out:
            skipped.append((label, "no results in logs/ — run this setting's benchmark first"))
        else:
            skipped.append((label, "plot failed (see output above)"))

    print("\n" + "=" * 60)
    for m in made:
        print(f"  ✓ {m}")
    for s, why in skipped:
        print(f"  – {s}: {why}")
    if not made:
        print("\nNo figures produced — run the settings' benchmarks first so their logs/ exist.")


if __name__ == "__main__":
    main()
