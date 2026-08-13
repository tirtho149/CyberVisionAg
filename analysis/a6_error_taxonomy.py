#!/usr/bin/env python3
"""Analysis 4 (task #6) — organ-aware error taxonomy from existing predictions.

For the clean Claude-sonnet errors (KB pipeline, k=8), categorize each error by
joining the KB registry's affected_parts (organ) for the TRUE vs PREDICTED disease:
  - same-organ look-alike : true & pred share >=1 affected organ (hard case; the
                            anatomical prior cannot separate them)
  - cross-organ           : disjoint organs (the organ prior SHOULD have pruned)
  - unknown-organ         : organ missing for true or pred in the KB
Also reports confident-error rate (conf>=0.8) and the top confusion pairs.
No inference.
"""
import csv, glob, json, os, re, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)


def norm(s): return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def crop_key(cropdir):  # Soybean_Diseases -> Soybean ; Mango_Leaf_Disease -> Mango_Leaf
    return re.sub(r"_Diseases?$", "", cropdir)


def load_organs():
    """(crop_key_norm, disease_norm) -> set(organs)."""
    organs = {}
    for f in glob.glob(os.path.join(HERE, "..", "disease_registry", "outputs", "*", "final_registry.json")):
        if ".bak" in f: continue
        crop = norm(os.path.basename(os.path.dirname(f)))
        try: d = json.load(open(f))
        except Exception: continue
        for dis in d.get("diseases", []):
            ap = dis.get("affected_parts", {})
            val = ap.get("value") if isinstance(ap, dict) else ap
            organs[(crop, norm(dis.get("disease_name", "")))] = set(norm(v) for v in (val or []))
    return organs


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "predictions.csv"))))
    errs = [r for r in rows if r["source"] == "SAGE" and r["model"] == "sonnet"
            and r["has_error"] == "False" and r["condition"] == "internet"
            and r["k"] == "8" and r["correct"] != "True"]
    organs = load_organs()
    ck = {}  # cache crop_key norm
    cats = collections.Counter(); confident = 0; conf_by_cat = collections.defaultdict(list)
    pairs = collections.Counter()
    for r in errs:
        crop = norm(crop_key(r["crop"]))
        t, p = norm(r["disease_true"]), norm(r["disease_pred"])
        ot, op = organs.get((crop, t)), organs.get((crop, p))
        if ot is None or op is None or not ot or not op:
            cat = "unknown-organ"
        elif ot & op:
            cat = "same-organ look-alike"
        else:
            cat = "cross-organ"
        cats[cat] += 1
        try:
            c = float(r["confidence"])
            if c >= 0.8: confident += 1
            conf_by_cat[cat].append(c)
        except (ValueError, TypeError): pass
        pairs[(r["crop"].split("_")[0], r["disease_true"], r["disease_pred"])] += 1

    n = len(errs)
    lines = [f"=== Error taxonomy (SAGE sonnet, internet KB, k=8): {n} errors ==="]
    for cat, c in cats.most_common():
        mc = np.mean(conf_by_cat[cat])*100 if conf_by_cat[cat] else float("nan")
        lines.append(f"  {cat:22} {c:4d}  ({100*c/n:4.1f}%)   mean conf {mc:.0f}%")
    lines.append(f"\nconfident errors (conf>=0.8): {confident}/{n} = {100*confident/n:.1f}%  "
                 f"<- wrong-but-confident")
    lines.append("\ntop 12 confusion pairs (crop: true -> pred):")
    for (cr, t, p), c in pairs.most_common(12):
        lines.append(f"  {c:3d}x  {cr}: {t} -> {p}")
    # error concentration
    top5 = sum(c for _, c in pairs.most_common(5))
    lines.append(f"\nerror concentration: top-5 confusion pairs = {top5}/{n} = {100*top5/n:.1f}% of errors")
    txt = "\n".join(lines); print(txt)
    open(os.path.join(OUT, "a6_error_taxonomy.txt"), "w").write(txt+"\n")
    print(f"\nwrote {OUT}/a6_error_taxonomy.txt")


if __name__ == "__main__":
    main()
