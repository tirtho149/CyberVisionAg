"""
Two-layer LLM-as-Judge for crop disease dataset quality.

LAYER 1 — Crop Validation (decision tree)
  Node A: Is this entry a real plant/organism at all?
    → NO  → INVALID (e.g. spreadsheet artefacts like "TOTAL")
    → YES → Node B
  Node B: Is it a cultivated crop used in agriculture or horticulture?
    → NO  → NON_CROP (wild plant, purely decorative, non-crop organism)
    → YES → Node C
  Node C: Is the crop name correctly spelled and standardized?
    → NO  → MISSPELLED / UNSTANDARDIZED
    → YES → VALID

LAYER 2 — Disease Label Quality (only for VALID / MISSPELLED crops)
  For each disease label:
    CORRECT / INCORRECT / QUESTIONABLE
  Plus: similar / duplicate group detection.

Uses `claude -p` headless subprocess — no API key needed.
Results stream live to disease_label_progress.txt.
Automatically resumes from saved results on re-run.
"""

import csv
import json
import re
import subprocess
import time
from pathlib import Path

CSV_PATH      = Path(__file__).parent / "crop_disease_registry.csv"
OUTPUT_PATH   = Path(__file__).parent / "disease_label_full_report.json"
PROGRESS_PATH = Path(__file__).parent / "disease_label_full_progress.txt"

# ── prompts ──────────────────────────────────────────────────────────────────

CROP_VALIDATION_PROMPT = """\
You are an expert agronomist. Apply the decision tree below to the crop name given.

DECISION TREE:
  Node A: Is this entry a real plant or plant-based crop (not a spreadsheet label, number, or non-plant)?
    → NO  → verdict: INVALID   (e.g. "TOTAL", "N/A", a row header)
    → YES → Node B
  Node B: Is it grown as an agricultural or horticultural crop (food, fibre, spice, timber, ornamental)?
    → NO  → verdict: NON_CROP  (purely wild plant, a pathogen name, not cultivated)
    → YES → Node C
  Node C: Is the common crop name spelled correctly and reasonably standardised?
    → NO  → verdict: MISSPELLED_OR_UNSTANDARDISED
    → YES → verdict: VALID

Return ONLY valid JSON (no markdown, no fences):
{
  "crop": "<exact input name>",
  "node_a": "YES" | "NO",
  "node_b": "YES" | "NO" | "N/A",
  "node_c": "YES" | "NO" | "N/A",
  "verdict": "VALID" | "INVALID" | "NON_CROP" | "MISSPELLED_OR_UNSTANDARDISED",
  "canonical_name": "<corrected / standard name, or same if VALID>",
  "category": "<one of: cereal, legume, fruit, vegetable, root_tuber, tree_crop, fiber_crop, oilseed, herb_spice, ornamental, beverage_crop, nut_crop, forage, other>",
  "notes": "<one sentence — only if something is unusual or needs clarification, else empty string>"
}

Crop to evaluate:
"""

DISEASE_JUDGE_PROMPT = """\
You are an expert plant pathologist. Quality-check disease labels for the crop below.

Return ONLY valid JSON (no markdown, no fences):
{
  "crop": "<crop name>",
  "disease_verdicts": [
    {
      "disease": "<label>",
      "verdict": "CORRECT" | "INCORRECT" | "QUESTIONABLE",
      "reason": "<one sentence>"
    }
  ],
  "similar_groups": [
    {
      "diseases": ["<label A>", "<label B>"],
      "reason": "<why these overlap or duplicate>"
    }
  ],
  "summary": "<2-3 sentence quality assessment>"
}

Verdict rules:
  CORRECT      — well-documented disease known to affect this crop.
  INCORRECT    — wrong crop; disease of a completely different plant; or entry is not a disease at all.
  QUESTIONABLE — real disease but label is vague, misspelled, names a pathogen genus instead of a
                 disease, refers to an insect pest or beneficial organism, or is an unusual
                 / unverified association with this crop.

"""

# ── helpers ───────────────────────────────────────────────────────────────────

def load_crop_disease_map(csv_path: Path) -> dict[str, list[str]]:
    crop_disease: dict[str, list[str]] = {}
    current_crop = ""
    current_disease = ""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            crop    = row["Crop"].strip()
            disease = row["Disease"].strip()
            if crop:    current_crop    = crop
            if disease: current_disease = disease
            if not current_crop or not current_disease:
                continue
            crop_disease.setdefault(current_crop, [])
            if current_disease not in crop_disease[current_crop]:
                crop_disease[current_crop].append(current_disease)
    return crop_disease


def call_claude(prompt: str, retries: int = 3) -> str:
    """
    Call `claude -p <prompt>` in plain-text mode.
    Returns the raw text response, or raises RuntimeError after all retries.
    """
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            ["claude", "--model", "claude-opus-4-7", "-p", prompt],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        # back off before retry
        if attempt < retries:
            time.sleep(4 * attempt)

    raise RuntimeError(
        f"claude CLI failed after {retries} retries. "
        f"stderr: {result.stderr.strip()[:200]}"
    )


def parse_json_from_text(text: str) -> dict:
    """Strip markdown fences and parse JSON; try partial extraction as fallback."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Grab first complete {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise


def write_progress(text: str) -> None:
    with open(PROGRESS_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")
        f.flush()


def log(text: str) -> None:
    print(text)
    write_progress(text)

# ── Layer 1: crop validation ──────────────────────────────────────────────────

def validate_crop(crop: str) -> dict:
    prompt = CROP_VALIDATION_PROMPT + crop
    try:
        raw = call_claude(prompt)
        result = parse_json_from_text(raw)
        result.setdefault("crop", crop)
        return result
    except Exception as e:
        return {"crop": crop, "verdict": "ERROR", "error": str(e), "parse_error": True}


def print_crop_validation(result: dict, idx: int, total: int) -> None:
    crop    = result.get("crop", "?")
    verdict = result.get("verdict", "ERROR")
    canon   = result.get("canonical_name", "")
    cat     = result.get("category", "")
    notes   = result.get("notes", "")
    node_a  = result.get("node_a", "?")
    node_b  = result.get("node_b", "?")
    node_c  = result.get("node_c", "?")

    lines = [
        f"\n{'─'*70}",
        f"[{idx}/{total}]  CROP: {crop}",
        f"{'─'*70}",
        f"  Node A (real plant?)     : {node_a}",
        f"  Node B (cultivated crop?): {node_b}",
        f"  Node C (name correct?)   : {node_c}",
        f"  Verdict   : {verdict}",
        f"  Category  : {cat}",
    ]
    if canon and canon != crop:
        lines.append(f"  Canonical : {canon}")
    if notes:
        lines.append(f"  Notes     : {notes}")
    output = "\n".join(lines)
    log(output)

# ── Layer 2: disease label check ──────────────────────────────────────────────

def judge_diseases(crop: str, diseases: list[str]) -> dict:
    disease_list = "\n".join(f"- {d}" for d in diseases)
    prompt = DISEASE_JUDGE_PROMPT + f"Crop: {crop}\n\nDiseases:\n{disease_list}"
    try:
        raw = call_claude(prompt)
        result = parse_json_from_text(raw)
        result.setdefault("crop", crop)
        return result
    except Exception as e:
        return {"crop": crop, "error": str(e), "parse_error": True}


def print_disease_report(result: dict, idx: int, total: int) -> None:
    crop = result.get("crop", "?")
    lines = [f"\n{'='*70}", f"[{idx}/{total}]  DISEASES — CROP: {crop}", "=" * 70]

    if result.get("parse_error"):
        lines.append(f"  [ERROR] {result.get('error') or result.get('raw_response','')[:300] or '(empty)'}")
        log("\n".join(lines)); return

    verdicts     = result.get("disease_verdicts", [])
    incorrect    = [v for v in verdicts if v["verdict"] == "INCORRECT"]
    questionable = [v for v in verdicts if v["verdict"] == "QUESTIONABLE"]
    correct      = [v for v in verdicts if v["verdict"] == "CORRECT"]

    lines += [
        f"  Total: {len(verdicts)}  |  CORRECT: {len(correct)}  "
        f"QUESTIONABLE: {len(questionable)}  INCORRECT: {len(incorrect)}",
    ]

    if incorrect:
        lines.append("\n  INCORRECT:")
        for v in incorrect:
            lines.append(f"    x  {v['disease']}: {v['reason']}")

    if questionable:
        lines.append("\n  QUESTIONABLE:")
        for v in questionable:
            lines.append(f"    ?  {v['disease']}: {v['reason']}")

    for g in result.get("similar_groups", []):
        lines.append(f"\n  ~ SIMILAR: {' | '.join(g['diseases'])}")
        lines.append(f"    {g['reason']}")

    if result.get("summary"):
        lines.append(f"\n  SUMMARY: {result['summary']}")

    log("\n".join(lines))

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    PROGRESS_PATH.write_text("", encoding="utf-8")

    header = "TWO-LAYER CROP DISEASE LABEL QUALITY REPORT\n" + "=" * 70
    log(header)

    log("Loading crop-disease data...")
    crop_disease_map = load_crop_disease_map(CSV_PATH)
    crops = sorted(crop_disease_map.keys())
    log(f"Found {len(crops)} crops, "
        f"{sum(len(v) for v in crop_disease_map.values())} unique disease labels.\n")

    # Load saved results for resume support
    saved: dict[str, dict] = {}
    if OUTPUT_PATH.exists():
        try:
            with open(OUTPUT_PATH, encoding="utf-8") as f:
                for r in json.load(f):
                    crop = r.get("crop", "")
                    if crop and not r.get("parse_error"):
                        saved[crop] = r
            if saved:
                log(f"Resuming: {len(saved)} crops already completed — skipping.\n")
        except Exception:
            pass

    all_results: list[dict] = []
    total = len(crops)

    for idx, crop in enumerate(crops, start=1):
        # ── skip if already done ──
        if crop in saved:
            all_results.append(saved[crop])
            print(f"[{idx}/{total}] {crop} — skipped (saved)")
            continue

        record: dict = {"crop": crop}

        # ── LAYER 1: crop validation ──
        log(f"\n{'#'*70}")
        log(f"# [{idx}/{total}]  {crop}")
        log(f"# LAYER 1 — Crop Validation")
        log(f"{'#'*70}")

        crop_val = validate_crop(crop)
        record["crop_validation"] = crop_val
        print_crop_validation(crop_val, idx, total)

        verdict = crop_val.get("verdict", "ERROR")

        # ── LAYER 2: disease checking (only for valid / misspelled crops) ──
        skip_disease = verdict in ("INVALID", "NON_CROP", "ERROR")

        if skip_disease:
            log(f"\n  → Skipping disease check (crop verdict: {verdict})")
            record["disease_check"] = {
                "skipped": True,
                "reason": f"Crop verdict is {verdict}",
            }
        else:
            log(f"\n# LAYER 2 — Disease Label Check")
            diseases = crop_disease_map[crop]
            print(f"  Judging {len(diseases)} diseases...", end=" ", flush=True)
            write_progress(f"  Judging {len(diseases)} diseases...")

            disease_result = judge_diseases(crop, diseases)
            record["disease_check"] = disease_result
            print("done")
            print_disease_report(disease_result, idx, total)

        all_results.append(record)

        # Save incrementally after every crop
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        time.sleep(1)   # gentle pacing

    log(f"\n\nFull JSON report saved to: {OUTPUT_PATH}")

    # ── aggregate statistics ──
    total_crops = len(all_results)
    invalid_crops = sum(
        1 for r in all_results
        if r.get("crop_validation", {}).get("verdict") in ("INVALID", "NON_CROP", "ERROR")
    )
    misnamed_crops = sum(
        1 for r in all_results
        if r.get("crop_validation", {}).get("verdict") == "MISSPELLED_OR_UNSTANDARDISED"
    )

    total_d = total_inc = total_q = total_sim = 0
    for r in all_results:
        dc = r.get("disease_check", {})
        if dc.get("skipped"):
            continue
        vs = dc.get("disease_verdicts", [])
        total_d   += len(vs)
        total_inc += sum(1 for v in vs if v["verdict"] == "INCORRECT")
        total_q   += sum(1 for v in vs if v["verdict"] == "QUESTIONABLE")
        total_sim += len(dc.get("similar_groups", []))

    summary = (
        "\n" + "=" * 70 + "\n"
        "OVERALL DATASET QUALITY SUMMARY\n" +
        "=" * 70 + "\n"
        f"  LAYER 1 — Crop Labels\n"
        f"    Total crops        : {total_crops}\n"
        f"    Invalid / Non-crop : {invalid_crops}\n"
        f"    Misspelled         : {misnamed_crops}\n"
        f"    Valid              : {total_crops - invalid_crops - misnamed_crops}\n"
        f"\n  LAYER 2 — Disease Labels (valid crops only)\n"
        f"    Total diseases     : {total_d}\n"
        f"    Incorrect          : {total_inc}  ({100*total_inc/max(total_d,1):.1f}%)\n"
        f"    Questionable       : {total_q}  ({100*total_q/max(total_d,1):.1f}%)\n"
        f"    Similar groups     : {total_sim}\n"
    )
    log(summary)


if __name__ == "__main__":
    main()
