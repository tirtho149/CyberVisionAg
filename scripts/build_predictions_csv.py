#!/usr/bin/env python3
"""Consolidate every per-image agentic prediction JSON into one flat predictions.csv.

Path schema:  <root>/open_agentic/<CROP>/<condition>/<model>/k<N>/<image_id>.json
  condition in {none, internet, few_shot}
  model     in {opus, sonnet, haiku, gemini-flash, gemini-*}
  k         in {0,1,4,8,16}

Stale/duplicate crop dirs (``*_old``, ``*.pre_*``, ``*_pre<digits>``) are skipped.
This reads ALL crops/models (Claude + Gemini) with a `source`+`model` column so
downstream analyses can filter to the clean Claude grid or a secondary slice.

Usage:  python scripts/build_predictions_csv.py [-o analysis/predictions.csv]
"""
import csv, json, os, re, argparse
from pathlib import Path

ROOTS = [
    Path("/work/mech-ai-scratch/tirtho/SAGE/results/open_agentic"),
    Path("/work/mech-ai-scratch/tirtho/spark/upstream/results/open_agentic"),
]
STALE = re.compile(r"(_old$|\.pre_|_pre\d+$)")

COLS = ["source", "crop", "condition", "model", "k", "image_id",
        "disease_true", "disease_pred", "correct", "confidence",
        "num_turns", "cost_usd", "duration_ms", "refs_viewed", "refs_n",
        "has_error", "error"]


def f(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def i(x):
    try: return int(float(x))
    except (TypeError, ValueError): return None


def as_bool(x):
    return str(x).strip().lower() in ("true", "1")


def refs_count(rv):
    if isinstance(rv, list): return len(rv)
    s = str(rv).strip()
    if s.startswith("["):
        try: return len(json.loads(s))
        except Exception: return None
    return i(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="/work/mech-ai-scratch/tirtho/SAGE/analysis/predictions.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    rows, skipped_stale, bad = [], set(), 0
    for root in ROOTS:
        if not root.is_dir():
            continue
        src = "SAGE" if "SAGE/results" in str(root) else "spark"
        for cropdir in sorted(root.iterdir()):
            if not cropdir.is_dir():
                continue
            if STALE.search(cropdir.name):
                skipped_stale.add(cropdir.name); continue
            for jf in cropdir.rglob("*.json"):
                if jf.name == "summary.json":
                    continue
                rel = jf.relative_to(cropdir).parts   # (condition, model, kN, file)
                if len(rel) != 4:
                    continue
                condition, model, kdir, _ = rel
                k = i(kdir[1:]) if kdir.lower().startswith("k") else None
                try:
                    d = json.load(open(jf))
                except Exception:
                    bad += 1; continue
                if not isinstance(d, dict):
                    bad += 1; continue
                err = d.get("error")
                err = None if str(err).strip() in ("None", "", "null") else str(err)
                rows.append({
                    "source": src,
                    "crop": cropdir.name,
                    "condition": condition,
                    "model": model,
                    "k": k,
                    "image_id": jf.stem,
                    "disease_true": d.get("ground_truth"),
                    "disease_pred": d.get("prediction"),
                    "correct": as_bool(d.get("correct")),
                    "confidence": f(d.get("confidence")),
                    "num_turns": i(d.get("num_turns")),
                    "cost_usd": f(d.get("cost_usd")),
                    "duration_ms": i(d.get("duration_ms")),
                    "refs_viewed": d.get("refs_viewed"),
                    "refs_n": refs_count(d.get("refs_viewed")),
                    "has_error": err is not None,
                    "error": err,
                })

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    # ---- summary ----
    print(f"wrote {len(rows):,} rows -> {a.out}")
    print(f"skipped stale dirs: {sorted(skipped_stale)}")
    print(f"unreadable json: {bad}")
    from collections import Counter
    print("\nby source:", dict(Counter(r["source"] for r in rows)))
    print("by model :", dict(Counter(r["model"] for r in rows)))
    print("by condition:", dict(Counter(r["condition"] for r in rows)))
    print("by k     :", dict(Counter(r["k"] for r in rows)))
    print("error rows:", sum(r["has_error"] for r in rows))
    print("\nrows per crop (SAGE source only):")
    for crop, n in sorted(Counter(r["crop"] for r in rows if r["source"] == "SAGE").items()):
        print(f"  {crop:34} {n}")


if __name__ == "__main__":
    main()
