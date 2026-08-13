#!/usr/bin/env python3
"""Analysis 1 — paired bootstrap significance + CIs for the KB effect.

Reads analysis/predictions.csv. Main-study slice: source=SAGE, model=sonnet,
has_error=False. For each crop and reference budget k, pairs the SAME test images
under condition=internet (KB) vs condition=none (no KB), then paired-bootstraps
Delta = acc(KB) - acc(no-KB). Reports Delta, 95% CI, P(Delta>0), Wilson CIs on the
raw accuracies, macro over the 4 headline crops, Holm correction across k, and the
headline full-pipeline (internet,k=8) vs baseline (none,k=0) contrast.
"""
import csv, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "predictions.csv")
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
B = 10000
MAIN4 = ["Soybean_Diseases", "Corn_Diseases", "Tomato_Diseases", "Mango_Leaf_Disease"]
rng = np.random.default_rng(42)


def load():
    rows = list(csv.DictReader(open(CSV)))
    return [r for r in rows if r["source"] == "SAGE" and r["model"] == "sonnet"
            and r["has_error"] == "False"]


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (100*(c-h), 100*(c+h))


def paired(rows, crop, kA, condA, kB, condB):
    """Return matched 0/1 arrays (a=condA/kA, b=condB/kB) over shared image_ids."""
    def idx(cond, k):
        return {r["image_id"]: (r["correct"] == "True")
                for r in rows if r["crop"] == crop and r["condition"] == cond and r["k"] == str(k)}
    A, Bd = idx(condA, kA), idx(condB, kB)
    ids = sorted(set(A) & set(Bd))
    return (np.array([A[i] for i in ids], float),
            np.array([Bd[i] for i in ids], float))


def boot_delta(a, b):
    n = len(a)
    if n == 0: return None
    ix = rng.integers(0, n, size=(B, n))
    db = a[ix].mean(1) - b[ix].mean(1)
    return db


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0]*len(pvals); m = len(pvals); prev = 0
    for rank, i in enumerate(order):
        v = min(1.0, (m - rank) * pvals[i]); v = max(v, prev); prev = v; adj[i] = v
    return adj


def main():
    rows = load()
    crops = sorted({r["crop"] for r in rows})
    ks = [0, 1, 4, 8, 16]

    # ---- per crop x k KB effect ----
    recs = []
    for crop in crops:
        for k in ks:
            a, b = paired(rows, crop, k, "internet", k, "none")   # a=KB, b=no-KB
            n = len(a)
            if n == 0: continue
            db = boot_delta(a, b)
            delta = 100*(a.mean() - b.mean())
            lo, hi = np.percentile(db, [2.5, 97.5])*100
            p_gt0 = float((db > 0).mean())
            p_two = 2*min((db > 0).mean(), (db < 0).mean())
            recs.append(dict(crop=crop, k=k, n=n,
                             acc_noKB=100*b.mean(), acc_KB=100*a.mean(),
                             wilson_noKB="[%.1f,%.1f]" % wilson(int(b.sum()), n),
                             wilson_KB="[%.1f,%.1f]" % wilson(int(a.sum()), n),
                             delta=delta, ci_lo=lo, ci_hi=hi,
                             p_gt0=p_gt0, p_two=min(1.0, p_two)))
    with open(os.path.join(OUT, "a1_kb_effect.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys())); w.writeheader()
        for r in recs: w.writerow({k: (("%.3f"%v) if isinstance(v, float) else v) for k, v in r.items()})

    # ---- macro, EVERY cell vs the SINGLE fixed baseline (none, k=0) ----
    # (paper convention: improvement over the per-crop k=0 no-KB baseline)
    print("=== Improvement over the FIXED baseline (none, k=0), macro over 4 headline crops, sonnet ===")
    print(f"{'cond':>9} {'k':>3} {'macroΔ':>8} {'95% CI':>18} {'P(Δ>0)':>8} {'p_two':>8} {'p_holm':>8}")
    macro_rows = []
    for cond in ("none", "internet"):
        for k in ks:
            if cond == "none" and k == 0:
                continue  # this IS the baseline (Δ=0 by definition)
            dbs, deltas = [], []
            for crop in MAIN4:
                a, b = paired(rows, crop, k, cond, 0, "none")   # vs fixed (none,k0)
                if len(a) == 0: continue
                dbs.append(boot_delta(a, b)); deltas.append(a.mean()-b.mean())
            if not dbs: continue
            macro_b = np.mean(np.vstack(dbs), axis=0)*100
            md = float(np.mean(deltas))*100
            lo, hi = np.percentile(macro_b, [2.5, 97.5])
            p_two = 2*min((macro_b > 0).mean(), (macro_b < 0).mean())
            macro_rows.append(dict(cond=cond, k=k, md=md, lo=lo, hi=hi,
                                   pg=float((macro_b > 0).mean()), pt=min(1.0, p_two)))
    padj = holm([r["pt"] for r in macro_rows])
    for r, pa in zip(macro_rows, padj):
        star = "*" if pa < 0.05 else (" " if pa >= 0.10 else "~")
        print(f"{r['cond']:>9} {r['k']:>3} {r['md']:>+7.1f} {('[%+.1f,%+.1f]'%(r['lo'],r['hi'])):>18} "
              f"{r['pg']:>8.3f} {r['pt']:>8.4f} {pa:>8.4f} {star}")

    # ---- headline: full pipeline (internet,k=8) vs baseline (none,k=0) ----
    print("\n=== Headline: full pipeline (KB, k=8) vs baseline (no-KB, k=0), sonnet ===")
    print(f"{'crop':16} {'base%':>7} {'full%':>7} {'Δ':>7} {'95% CI':>18} {'P(Δ>0)':>8} n")
    hdeltas = []
    for crop in crops:
        a, b = paired(rows, crop, 8, "internet", 0, "none")
        if len(a) == 0: continue
        db = boot_delta(a, b); delta = 100*(a.mean()-b.mean())
        lo, hi = np.percentile(db, [2.5, 97.5])*100
        tag = crop.split("_")[0]
        star = " *" if crop in MAIN4 else ""
        print(f"{tag:16} {100*b.mean():>7.1f} {100*a.mean():>7.1f} {delta:>+7.1f} "
              f"{('[%+.1f,%+.1f]'%(lo,hi)):>18} {float((db>0).mean()):>8.3f} {len(a)}{star}")
        if crop in MAIN4: hdeltas.append(a.mean()-b.mean())
    print(f"macro(4 headline crops) mean Δ = {100*np.mean(hdeltas):+.1f} pp")
    print(f"\nwrote {os.path.join(OUT,'a1_kb_effect.csv')}  (* = one of the 4 headline crops)")


if __name__ == "__main__":
    main()
