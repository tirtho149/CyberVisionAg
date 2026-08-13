#!/usr/bin/env python3
"""Analysis 3 — confidence calibration on already-saved predictions.

Reads analysis/predictions.csv (source=SAGE, model=sonnet, has_error=False, with a
numeric confidence). Computes ECE, MCE, Brier, mean confidence for correct vs
incorrect, AUROC(confidence -> correctness), P(correct|c>=0.9) vs P(correct|c<0.7),
a reliability-bin table, and a reliability-diagram PNG. No inference.
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)


def load():
    rows = list(csv.DictReader(open(os.path.join(HERE, "predictions.csv"))))
    conf, corr = [], []
    for r in rows:
        if r["source"] != "SAGE" or r["model"] != "sonnet" or r["has_error"] != "False":
            continue
        try: c = float(r["confidence"])
        except (ValueError, TypeError): continue
        if not (0.0 <= c <= 1.0): continue
        conf.append(c); corr.append(r["correct"] == "True")
    return np.array(conf), np.array(corr, float)


def auroc(score, label):
    # Mann-Whitney U / rank-based AUROC
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float); ranks[order] = np.arange(1, len(score)+1)
    # average ranks for ties
    s = score[order]; i = 0
    while i < len(s):
        j = i
        while j+1 < len(s) and s[j+1] == s[i]: j += 1
        if j > i:
            ranks[order[i:j+1]] = (i+1 + j+1) / 2.0
        i = j+1
    npos = label.sum(); nneg = len(label)-npos
    if npos == 0 or nneg == 0: return float("nan")
    return (ranks[label == 1].sum() - npos*(npos+1)/2) / (npos*nneg)


def ece_mce(conf, corr, nbins=15):
    edges = np.linspace(0, 1, nbins+1)
    ece = mce = 0.0; rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        n = m.sum()
        if n == 0:
            rows.append((lo, hi, 0, float("nan"), float("nan"))); continue
        acc = corr[m].mean(); cf = conf[m].mean(); gap = abs(acc-cf)
        ece += n/len(conf)*gap; mce = max(mce, gap)
        rows.append((lo, hi, int(n), cf, acc))
    return ece, mce, rows


def main():
    conf, corr = load()
    n = len(conf); acc = corr.mean()
    brier = np.mean((conf-corr)**2)
    ece, mce, bins = ece_mce(conf, corr)
    auc = auroc(conf, corr)
    mc_cor, mc_inc = conf[corr == 1].mean(), conf[corr == 0].mean()
    hi = conf >= 0.9; lo = conf < 0.7
    p_hi = corr[hi].mean() if hi.sum() else float("nan")
    p_lo = corr[lo].mean() if lo.sum() else float("nan")

    lines = [
        "=== Confidence calibration (SAGE, sonnet, clean) ===",
        f"n={n}  overall accuracy={acc*100:.1f}%  mean confidence={conf.mean()*100:.1f}%",
        f"ECE={ece*100:.2f}%   MCE={mce*100:.2f}%   Brier={brier:.4f}",
        f"AUROC(confidence -> correctness) = {auc:.3f}",
        f"mean confidence | correct   = {mc_cor*100:.1f}%",
        f"mean confidence | incorrect = {mc_inc*100:.1f}%   (gap {100*(mc_cor-mc_inc):+.1f} pp)",
        f"P(correct | conf>=0.9) = {p_hi*100:.1f}%  (n={int(hi.sum())})",
        f"P(correct | conf< 0.7) = {p_lo*100:.1f}%  (n={int(lo.sum())})",
        "",
        "reliability bins:  [lo,hi]   n   conf%   acc%   gap",
    ]
    for lo_, hi_, nb, cf, ac in bins:
        if nb == 0: continue
        lines.append(f"  ({lo_:.2f},{hi_:.2f}]  {nb:5d}  {cf*100:5.1f}  {ac*100:5.1f}  {abs(ac-cf)*100:+5.1f}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "a3_calibration.txt"), "w").write(txt+"\n")

    # reliability diagram
    xs = [(l+h)/2 for l, h, nb, cf, ac in bins if nb]
    accs = [ac for l, h, nb, cf, ac in bins if nb]
    ns = [nb for l, h, nb, cf, ac in bins if nb]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect")
    ax.bar(xs, accs, width=1/15*0.9, alpha=0.7, edgecolor="k", label="accuracy")
    for x, a_, nb in zip(xs, accs, ns):
        ax.text(x, a_+0.02, str(nb), ha="center", va="bottom", fontsize=6)
    ax.set_xlabel("confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_title(f"Reliability (ECE={ece*100:.1f}%, AUROC={auc:.2f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "a3_reliability.png"), dpi=150)
    print(f"\nwrote {OUT}/a3_calibration.txt and a3_reliability.png")


if __name__ == "__main__":
    main()
