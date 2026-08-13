#!/usr/bin/env python3
"""One-command reproduction of every quantitative analysis in the SAGE rebuttal.

Rebuilds predictions.csv from the saved per-image JSONs, then runs each analysis
(A1 bootstrap, A2 k-curve, A3 calibration, A6 error taxonomy, A7 cost/Pareto,
A8 KB-quality, A9 dataset stats, A10 stage retention). No model inference, no
GPU: everything is computed from already-saved outputs. Results land in
analysis/out/.  Usage:  python analysis/reproduce_all.py
"""
import os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = [
    ("build predictions.csv", ["scripts/build_predictions_csv.py"]),
    ("A1 paired bootstrap + CIs", ["analysis/a1_bootstrap.py"]),
    ("A2 k-curve uncertainty", ["analysis/a2_kcurve.py"]),
    ("A3 calibration", ["analysis/a3_calibration.py"]),
    ("A6 error taxonomy", ["analysis/a6_error_taxonomy.py"]),
    ("A7 cost / Pareto", ["analysis/a7_cost_pareto.py"]),
    ("A8 KB-quality vs accuracy", ["analysis/a8_kb_quality.py"]),
    ("A9 dataset characterization", ["analysis/a9_dataset_stats.py"]),
    ("A10 stage-wise retention", ["analysis/a10_stage_retention.py"]),
]


def main():
    fails = []
    for name, cmd in STEPS:
        print(f"\n{'='*70}\n>>> {name}\n{'='*70}")
        r = subprocess.run([sys.executable, *cmd], cwd=ROOT)
        if r.returncode != 0:
            fails.append(name)
    print(f"\n{'='*70}\nDONE. outputs in analysis/out/")
    if fails:
        print("FAILED:", fails); sys.exit(1)


if __name__ == "__main__":
    main()
