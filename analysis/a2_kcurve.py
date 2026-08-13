#!/usr/bin/env python3
"""Analysis 2 — is the k-curve actually monotonic, or sampling noise?

Paired bootstrap on consecutive reference-budget transitions (0->1, 1->4, 4->8,
8->16) for the KB pipeline (SAGE, sonnet, internet). Pairs the SAME images at both
budgets. If most transitions' CIs cross 0, "more references" is NOT reliably
monotonic -- a benchmark finding rather than a flaw.
"""
import csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
B = 10000
MAIN4 = ["Soybean_Diseases", "Corn_Diseases", "Tomato_Diseases", "Mango_Leaf_Disease"]
rng = np.random.default_rng(42)


def load(cond):
    rows = list(csv.DictReader(open(os.path.join(HERE, "predictions.csv"))))
    return [r for r in rows if r["source"] == "SAGE" and r["model"] == "sonnet"
            and r["has_error"] == "False" and r["condition"] == cond]


def paired(rows, crop, kHi, kLo):
    def idx(k):
        return {r["image_id"]: (r["correct"] == "True")
                for r in rows if r["crop"] == crop and r["k"] == str(k)}
    A, Bd = idx(kHi), idx(kLo)
    ids = sorted(set(A) & set(Bd))
    return np.array([A[i] for i in ids], float), np.array([Bd[i] for i in ids], float)


def boot(a, b):
    n = len(a)
    ix = rng.integers(0, n, size=(B, n))
    return (a[ix].mean(1) - b[ix].mean(1))*100


def main():
    for cond in ("internet", "none"):
        rows = load(cond)
        print(f"\n=== k-curve transitions (SAGE sonnet, {cond} KB), macro over 4 headline crops ===")
        print(f"{'transition':>12} {'macroΔ':>8} {'95% CI':>18} {'P(Δ>0)':>8}  monotonic?")
        recs = []
        for kLo, kHi in [(0, 1), (1, 4), (4, 8), (8, 16)]:
            dbs, deltas = [], []
            for crop in MAIN4:
                a, b = paired(rows, crop, kHi, kLo)
                if len(a) == 0: continue
                dbs.append(boot(a, b)); deltas.append(100*(a.mean()-b.mean()))
            if not dbs: continue
            mb = np.mean(np.vstack(dbs), axis=0); md = float(np.mean(deltas))
            lo, hi = np.percentile(mb, [2.5, 97.5])
            sig = "yes" if (lo > 0 or hi < 0) else "no (CI crosses 0)"
            print(f"{f'k{kLo}->k{kHi}':>12} {md:>+7.1f} {('[%+.1f,%+.1f]'%(lo,hi)):>18} "
                  f"{float((mb>0).mean()):>8.3f}  {sig}")
            recs.append(dict(cond=cond, trans=f"k{kLo}->k{kHi}", delta=md, lo=lo, hi=hi,
                             p_gt0=float((mb > 0).mean()), sig=sig))
        if cond == "internet":
            with open(os.path.join(OUT, "a2_kcurve.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(recs[0].keys())); w.writeheader()
                for r in recs: w.writerow({k: (("%.3f"%v) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nwrote {OUT}/a2_kcurve.csv")


if __name__ == "__main__":
    main()
