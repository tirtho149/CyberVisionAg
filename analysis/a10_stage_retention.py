"""Analysis 8 (task #10) — stage-wise candidate retention from saved traces.

Parses the detailed agent trace logs (`*_log.json`, trace[] of tool calls) to
measure whether the TRUE disease survives each narrowing stage of the pipeline:
  initial      -> present in list_dataset_classes (trivially ~100%)
  symptom-read -> agent called read_symptom_description for the true disease
  reference    -> agent called get_reference_image for the true disease
  final        -> predicted correctly
This is a *retention/filtering* view (does the pipeline keep the right answer
in play), not a full predictive ablation. Sample = whatever traces were saved.
"""
import glob, json, os, re, collections

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)


def norm(s): return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def names_in(inp):
    """Collect normalized string values from a tool-call input dict."""
    out = set()
    if isinstance(inp, dict):
        for v in inp.values():
            if isinstance(v, str): out.add(norm(v))
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, str): out.add(norm(x))
    elif isinstance(inp, str):
        out.add(norm(inp))
    return out


def main():
    logs = glob.glob(os.path.join(HERE, "..", "results-report", "**", "*_log.json"), recursive=True)
    n = 0
    stage = collections.Counter()          # how many traces retain truth at each stage
    used_symptom = used_ref = 0            # traces where those tools appeared at all
    for f in logs:
        try: d = json.load(open(f))
        except Exception: continue
        gt = norm(d.get("ground_truth", ""))
        if not gt: continue
        trace = d.get("trace", [])
        sym, ref = set(), set()
        saw_sym = saw_ref = False
        for s in trace:
            tool = s.get("tool"); inp = s.get("input", {})
            if tool == "read_symptom_description":
                saw_sym = True; sym |= names_in(inp)
            elif tool == "get_reference_image":
                saw_ref = True; ref |= names_in(inp)
        n += 1
        stage["initial"] += 1                                   # truth is in the class list by construction
        if saw_sym: used_symptom += 1
        if saw_ref: used_ref += 1
        # retention: truth appears (exact or as substring) in the stage's queried names
        def hit(S): return any(gt == x or gt in x or x in gt for x in S) if S else False
        if hit(sym): stage["symptom-read"] += 1
        if hit(ref): stage["reference"] += 1
        if d.get("prediction") and norm(d["prediction"]) == gt: stage["final-correct"] += 1

    lines = [f"=== Stage-wise candidate retention ({n} saved traces) ==="]
    lines.append(f"traces invoking read_symptom_description: {used_symptom}/{n}")
    lines.append(f"traces invoking get_reference_image      : {used_ref}/{n}")
    lines.append("\ntrue disease retained at stage:")
    for st in ("initial", "symptom-read", "reference", "final-correct"):
        c = stage.get(st, 0)
        lines.append(f"  {st:14} {c:4d}/{n}  ({100*c/max(n,1):5.1f}%)")
    lines.append("\nReading: the drop from 'initial' to 'symptom-read'/'reference' is the "
                 "narrowing cost (true answer dropped before the final compare); the drop to "
                 "'final-correct' among retained cases is the discrimination cost.")
    lines.append("CAVEAT: only traces that were saved to results-report/ are covered; "
                 "this is a retention view, not a predictive ablation.")
    txt = "\n".join(lines); print(txt)
    open(os.path.join(OUT, "a10_stage_retention.txt"), "w").write(txt+"\n")
    print(f"\nwrote {OUT}/a10_stage_retention.txt")


if __name__ == "__main__":
    main()
