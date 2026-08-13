#!/usr/bin/env python3
"""Analysis 6 (task #8) — does KB quality relate to downstream accuracy?

No expert-audit labels survive (see REVISION_PLAN.md), so we use the registry's
self-reported per-disease `confidence` (high/medium/low) and `num_sources` as
quality proxies. Join to the clean Claude-sonnet KB predictions (internet) and
compare accuracy across quality tiers. Descriptive only: KB confidence is
self-reported, tiers are unbalanced, so this is association, not causation.
"""
import csv, glob, json, os, re, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)


def norm(s): return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")
def crop_key(c): return norm(re.sub(r"_Diseases?$", "", c))


def load_quality():
    q = {}
    for f in glob.glob(os.path.join(HERE, "..", "disease_registry", "outputs", "*", "final_registry.json")):
        if ".bak" in f: continue
        crop = norm(os.path.basename(os.path.dirname(f)))
        try: d = json.load(open(f))
        except Exception: continue
        for dis in d.get("diseases", []):
            q[(crop, norm(dis.get("disease_name", "")))] = (
                str(dis.get("confidence", "?")).lower(), dis.get("num_sources"))
    return q


def main():
    q = load_quality()
    rows = list(csv.DictReader(open(os.path.join(HERE, "predictions.csv"))))
    rows = [r for r in rows if r["source"] == "SAGE" and r["model"] == "sonnet"
            and r["has_error"] == "False" and r["condition"] == "internet"]
    by_tier = collections.defaultdict(lambda: [0, 0])   # tier -> [correct, total]
    by_src = collections.defaultdict(lambda: [0, 0])
    matched = 0
    for r in rows:
        key = (crop_key(r["crop"]), norm(r["disease_true"]))
        if key not in q: continue
        matched += 1
        tier, ns = q[key]
        ok = r["correct"] == "True"
        by_tier[tier][0] += ok; by_tier[tier][1] += 1
        bucket = ">=2 sources" if (ns or 0) >= 2 else "1 source" if ns == 1 else "0/unknown"
        by_src[bucket][0] += ok; by_src[bucket][1] += 1

    lines = [f"=== KB quality vs accuracy (SAGE sonnet, internet; {matched} joined predictions) ==="]
    lines.append("by KB self-reported confidence of the TRUE disease:")
    for tier in ("high", "medium", "low"):
        c, n = by_tier.get(tier, [0, 0])
        if n: lines.append(f"  {tier:7} acc {100*c/n:5.1f}%   (n={n})")
    lines.append("by number of source citations for the TRUE disease:")
    for b in (">=2 sources", "1 source", "0/unknown"):
        c, n = by_src.get(b, [0, 0])
        if n: lines.append(f"  {b:12} acc {100*c/n:5.1f}%   (n={n})")
    # simple odds ratio high vs low
    ch, nh = by_tier.get("high", [0, 0]); cl, nl = by_tier.get("low", [0, 0])
    if nh and nl and ch < nh and cl < nl:
        or_ = (ch/(nh-ch)) / (cl/(nl-cl)) if (nh-ch) and cl else float("nan")
        lines.append(f"\nodds(correct) high-conf vs low-conf KB = {or_:.2f}x")
    lines.append("\nNOTE: confidence is the registry's self-report, not an expert label; "
                 "tiers are unbalanced. Association only.")
    txt = "\n".join(lines); print(txt)
    open(os.path.join(OUT, "a8_kb_quality.txt"), "w").write(txt+"\n")
    print(f"\nwrote {OUT}/a8_kb_quality.txt")


if __name__ == "__main__":
    main()
