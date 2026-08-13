#!/usr/bin/env python3
"""Analysis 9 (task #11) — dataset characterization (no model calls).

Image-level stats from the provenance sheet (crop, disease, source, filename):
counts, images/crop + images/class distributions, long tail, Gini, source
diversity. KB-field stats from the disease-registry JSONs (organ distribution,
source-citation counts, confidence mix, look-alike coverage).
CAVEAT: the provenance CSV is at/near Excel's 1,048,576-row cap and may be
truncated; counts are a lower bound if so.
"""
import csv, glob, json, os, re, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
PROV = "/work/mech-ai-scratch/tirtho/CyAg/crop_disease_registry_sheet3_image_sources.csv"


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    if n == 0 or x.sum() == 0: return float("nan")
    return (2*np.sum((np.arange(1, n+1))*x)/(n*x.sum())) - (n+1)/n


def main():
    lines = ["=== Dataset characterization ==="]
    # ---- provenance sheet ----
    per_crop = collections.Counter(); per_class = collections.Counter()
    sources = collections.Counter(); crops_dis = collections.defaultdict(set)
    n = 0
    with open(PROV, newline="") as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            crop = row.get("Crop"); dis = row.get("Disease"); src = row.get("Source")
            if not crop: continue
            n += 1
            per_crop[crop] += 1; per_class[(crop, dis)] += 1
            sources[src] += 1; crops_dis[crop].add(dis)
    lines.append(f"[provenance sheet]  images={n:,}  crops={len(per_crop)}  "
                 f"disease-classes={len(per_class)}  sources={len(sources)}")
    if n >= 1048575:
        lines.append("  !! WARNING: at Excel 1,048,575-row cap -> counts are a LOWER BOUND (truncated).")
    ic = np.array(list(per_crop.values())); cc = np.array(list(per_class.values()))
    lines.append(f"  images/crop : min {ic.min()}  median {int(np.median(ic))}  "
                 f"max {ic.max()}  mean {ic.mean():.0f}  Gini {gini(ic):.2f}")
    lines.append(f"  images/class: min {cc.min()}  median {int(np.median(cc))}  "
                 f"max {cc.max()}  mean {cc.mean():.0f}  Gini {gini(cc):.2f}")
    dpc = np.array([len(v) for v in crops_dis.values()])
    lines.append(f"  diseases/crop: min {dpc.min()}  median {int(np.median(dpc))}  max {dpc.max()}")
    for thr in (20, 50, 100):
        lines.append(f"  long tail: {100*np.mean(cc < thr):.1f}% of classes have < {thr} images")
    lines.append("  top sources: " + ", ".join(f"{s}({c:,})" for s, c in sources.most_common(6)))

    # ---- registries (study crops) ----
    organ = collections.Counter(); conf = collections.Counter()
    nsrc = []; nlook = []; ndis = 0; ncrop = 0
    for f in glob.glob(os.path.join(HERE, "..", "disease_registry", "outputs", "*", "final_registry.json")):
        if ".bak" in f: continue
        try: d = json.load(open(f))
        except Exception: continue
        ncrop += 1
        for dis in d.get("diseases", []):
            ndis += 1
            ap = dis.get("affected_parts", {}); val = ap.get("value") if isinstance(ap, dict) else ap
            for o in (val or []): organ[re.sub(r"\s+", "_", str(o))] += 1
            conf[dis.get("confidence", "?")] += 1
            if dis.get("num_sources") is not None: nsrc.append(dis["num_sources"])
            vs = dis.get("visual_symptoms", {})
            la = vs.get("look_alikes", {}) if isinstance(vs, dict) else {}
            lav = la.get("value") if isinstance(la, dict) else la
            nlook.append(len(lav or []))
    lines.append(f"\n[disease registries]  crops={ncrop}  diseases={ndis}  "
                 f"avg sources/disease={np.mean(nsrc):.1f}" if nsrc else "\n[disease registries] none")
    lines.append(f"  organ coverage: " + ", ".join(f"{o}({c})" for o, c in organ.most_common(8)))
    lines.append(f"  KB confidence mix: " + ", ".join(f"{k}={v}({100*v/max(ndis,1):.0f}%)" for k, v in conf.most_common()))
    if nlook: lines.append(f"  look-alikes: {100*np.mean(np.array(nlook)>0):.0f}% of diseases list >=1; "
                           f"mean {np.mean(nlook):.1f}")

    txt = "\n".join(lines); print(txt)
    open(os.path.join(OUT, "a9_dataset_stats.txt"), "w").write(txt+"\n")
    print(f"\nwrote {OUT}/a9_dataset_stats.txt")


if __name__ == "__main__":
    main()
