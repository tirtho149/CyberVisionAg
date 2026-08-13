#!/usr/bin/env python3
"""Analysis 7 — cost / token / latency / Pareto vs reference budget k.

Reads analysis/predictions.csv (source=SAGE, model=sonnet, has_error=False,
condition=internet — the KB pipeline). Per k reports accuracy, avg num_turns,
avg cost_usd, avg latency, marginal ΔAcc/ΔCost between consecutive budgets, and
flags the Pareto-efficient knee. No inference.
"""
import csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
MAIN4 = {"Soybean_Diseases", "Corn_Diseases", "Tomato_Diseases", "Mango_Leaf_Disease"}


def load(cond="internet"):
    rows = list(csv.DictReader(open(os.path.join(HERE, "predictions.csv"))))
    out = []
    for r in rows:
        if (r["source"] == "SAGE" and r["model"] == "sonnet" and r["has_error"] == "False"
                and r["condition"] == cond and r["crop"] in MAIN4):
            out.append(r)
    return out


def fnum(r, key):
    try: return float(r[key])
    except (ValueError, TypeError): return np.nan


def main():
    rows = load("internet")
    ks = [0, 1, 4, 8, 16]
    recs = []
    for k in ks:
        s = [r for r in rows if r["k"] == str(k)]
        if not s: continue
        acc = 100*np.mean([r["correct"] == "True" for r in s])
        turns = np.nanmean([fnum(r, "num_turns") for r in s])
        cost = np.nanmean([fnum(r, "cost_usd") for r in s])
        lat = np.nanmean([fnum(r, "duration_ms") for r in s])/1000.0
        recs.append(dict(k=k, n=len(s), acc=acc, turns=turns, cost=cost, lat=lat))

    # marginal efficiency vs previous k
    print("=== Cost / latency / Pareto (SAGE, sonnet, internet KB, 4 headline crops) ===")
    print(f"{'k':>3} {'n':>5} {'acc%':>7} {'turns':>6} {'$/img':>8} {'sec/img':>8} "
          f"{'ΔAcc':>6} {'Δ$':>8} {'ΔAcc/Δ$':>9}")
    prev = None
    for r in recs:
        if prev is None:
            dacc = dcost = eff = np.nan
        else:
            dacc = r["acc"]-prev["acc"]; dcost = r["cost"]-prev["cost"]
            eff = dacc/dcost if dcost else np.nan
        print(f"{r['k']:>3} {r['n']:>5} {r['acc']:>7.1f} {r['turns']:>6.1f} "
              f"{r['cost']:>8.4f} {r['lat']:>8.1f} "
              f"{('' if np.isnan(dacc) else '%+.1f'%dacc):>6} "
              f"{('' if np.isnan(dcost) else '%+.4f'%dcost):>8} "
              f"{('' if np.isnan(eff) else '%.0f'%eff):>9}")
        r["dacc"], r["dcost"], r["eff"] = dacc, dcost, eff
        prev = r

    # Pareto knee = best accuracy-per-dollar among k>0 with positive marginal acc
    cand = [r for r in recs if r["k"] > 0 and not np.isnan(r["eff"])]
    if cand:
        knee = max(cand, key=lambda r: (r["dacc"] > 0, r["eff"]))
        best = max(recs, key=lambda r: r["acc"])
        print(f"\nPeak accuracy: k={best['k']} ({best['acc']:.1f}%, ${best['cost']:.3f}/img)")
        print(f"Pareto knee (best ΔAcc/Δ$ with positive gain): k={knee['k']} "
              f"({knee['acc']:.1f}%, ${knee['cost']:.3f}/img)")

    with open(os.path.join(OUT, "a7_cost_pareto.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["k", "n", "acc", "turns", "cost", "lat", "dacc", "dcost", "eff"])
        w.writeheader()
        for r in recs:
            w.writerow({k: (("%.4f" % v) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nwrote {OUT}/a7_cost_pareto.csv")


if __name__ == "__main__":
    main()
