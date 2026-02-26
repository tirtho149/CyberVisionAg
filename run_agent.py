#!/usr/bin/env python3
"""
Agent evaluation matching AGENT_FLOW.md exactly.

Flow per image:
  Turn 1 — Send exact prompt + target image → initial prediction + needs_reference flag
  Turn N — If needs_reference=true: inject reference image for top candidate → re-reason
            Loop until needs_reference=false OR max_reference_turns reached
  Final  — Always runs, forces clean prediction JSON
  Judge  — LLM-as-judge scores confidence calibration immediately after prediction

Reference images are sourced from Curated_Local_Dataset/train/<category>/<class>/
"""

import os
import json
import base64
import re
from pathlib import Path
from datetime import datetime
from dotenv import dotenv_values
import requests

# Load environment variables
env_file = Path(__file__).parent / ".env"
env_vars = dotenv_values(env_file)
for key, value in env_vars.items():
    if value:
        os.environ[key] = value

# Configuration
CURATED_DIR          = Path(__file__).parent / "Curated_Local_Dataset"
TRAIN_DIR            = CURATED_DIR / "train"
TEST_DIR             = CURATED_DIR / "test"
RESULTS_DIR          = Path(__file__).parent / "results" / "agent"
SYMPTOMS_FILE        = Path(__file__).parent / "disease_symptoms_crop_wise.md"

MODEL               = "claude-sonnet-4-6"
MAX_TOKENS          = 4096
MAX_REFERENCE_TURNS = 3

SYSTEM_PROMPT = """You are an expert plant pathologist classifying plant diseases from images.
You will receive a target image and must classify it into one of the provided disease classes.
Always reason carefully before predicting. Never refuse to predict."""

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing AI plant disease classification agents.
Evaluate ONLY confidence calibration: did the agent's stated confidence match the actual outcome?

Calibration verdicts:
  WELL_CALIBRATED  — confidence matched outcome (high+correct or low+wrong)
  OVERCONFIDENT    — stated high confidence (>0.7) but was incorrect
  UNDERCONFIDENT   — stated low confidence (<0.4) but was correct
  INCONSISTENT     — confidence didn't reflect the quality of reasoning shown

Respond ONLY with valid JSON matching the schema in the prompt."""


# ── Utilities ──────────────────────────────────────────────────────────────────

def load_symptoms() -> str:
    """Load the full crop-wise symptoms markdown file."""
    with open(SYMPTOMS_FILE, "r") as f:
        return f.read()


def parse_symptoms_by_crop(symptoms_text: str) -> dict:
    """
    Parse disease_symptoms_crop_wise.md into a nested dict:
        {
          "Corn Diseases": {
              "Gray_leaf_spot": "Symptoms text...",
              "Common_rust":    "Symptoms text...",
              ...
          },
          ...
        }

    Crop headers    -> ## Crop: <name>
    Disease headers -> ### <Disease Name>
    Symptoms block  -> text between **Symptoms:** and **Reference Images:**
    """
    parsed: dict = {}
    current_crop: str = ""
    current_disease: str = ""
    collecting_symptoms: bool = False
    symptom_lines: list = []

    for line in symptoms_text.splitlines():
        # Crop header
        crop_match = re.match(r"^##\s+Crop:\s+(.+)$", line)
        if crop_match:
            if current_crop and current_disease and symptom_lines:
                parsed.setdefault(current_crop, {})[current_disease] = " ".join(symptom_lines).strip()
            current_crop = crop_match.group(1).strip()
            current_disease = ""
            symptom_lines = []
            collecting_symptoms = False
            parsed.setdefault(current_crop, {})
            continue

        # Disease header
        disease_match = re.match(r"^###\s+(.+)$", line)
        if disease_match:
            if current_crop and current_disease and symptom_lines:
                parsed.setdefault(current_crop, {})[current_disease] = " ".join(symptom_lines).strip()
            current_disease = disease_match.group(1).strip().replace(" ", "_")
            symptom_lines = []
            collecting_symptoms = False
            continue

        # Start collecting after **Symptoms:**
        if line.strip() == "**Symptoms:**":
            collecting_symptoms = True
            symptom_lines = []
            continue

        # Stop collecting at **Reference Images:**
        if line.strip() == "**Reference Images:**":
            collecting_symptoms = False
            continue

        # Accumulate symptom text (skip list lines)
        if collecting_symptoms and line.strip() and not line.startswith("-"):
            symptom_lines.append(line.strip())

    # Flush the last disease section
    if current_crop and current_disease and symptom_lines:
        parsed.setdefault(current_crop, {})[current_disease] = " ".join(symptom_lines).strip()

    return parsed


def extract_symptoms_for_dataset(symptoms_text: str, dataset_name: str,
                                  expected_classes: list) -> str:
    """
    Extract ONLY the crop section matching `dataset_name` and within that
    only the diseases listed in `expected_classes`.

    Returns a compact, prompt-ready string:

        === KNOWLEDGE BASE: Corn Diseases ===

        [Common_rust]
        Symptoms: ...

        [Gray_leaf_spot]
        Symptoms: ...

    Falls back to the full text if no matching crop section is found.
    """
    all_symptoms = parse_symptoms_by_crop(symptoms_text)

    # Find the best matching crop key
    dataset_normalised = dataset_name.replace("_", " ").lower()
    matched_crop = None
    for crop_key in all_symptoms:
        if crop_key.lower() == dataset_normalised:
            matched_crop = crop_key
            break
    if not matched_crop:
        for crop_key in all_symptoms:
            if dataset_normalised in crop_key.lower() or crop_key.lower() in dataset_normalised:
                matched_crop = crop_key
                break

    if not matched_crop:
        return symptoms_text  # safe fallback

    crop_symptoms = all_symptoms[matched_crop]

    lines = [f"=== KNOWLEDGE BASE: {matched_crop} ===\n"]
    matched_any = False
    for cls in expected_classes:
        cls_key = cls.replace(" ", "_")
        symptom_text = crop_symptoms.get(cls_key) or crop_symptoms.get(cls)
        if not symptom_text:
            lower_map = {k.lower(): v for k, v in crop_symptoms.items()}
            symptom_text = (lower_map.get(cls_key.lower())
                            or lower_map.get(cls.lower()))
        if symptom_text:
            lines.append(f"[{cls}]")
            lines.append(f"Symptoms: {symptom_text}\n")
            matched_any = True

    if not matched_any:
        for disease, text in crop_symptoms.items():
            lines.append(f"[{disease}]")
            lines.append(f"Symptoms: {text}\n")

    return "\n".join(lines)


MAX_IMAGE_BYTES = 4 * 1024 * 1024   # 4 MB — stay comfortably under the 5 MB API limit


def encode_image(image_path: str, max_bytes: int = MAX_IMAGE_BYTES) -> tuple:
    """
    Encode image to base64.  If the raw file exceeds `max_bytes`, progressively
    resize / recompress it (using Pillow) until it fits — avoiding the 5 MB hard
    limit enforced by the Anthropic API.
    """
    ext = Path(image_path).suffix.lower()
    media_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".webp": "image/webp",
    }.get(ext, "image/jpeg")

    raw_bytes = Path(image_path).read_bytes()

    if len(raw_bytes) <= max_bytes:
        return base64.standard_b64encode(raw_bytes).decode("utf-8"), media_type

    # File is too large — resize/recompress with Pillow
    try:
        from PIL import Image
        import io

        img = Image.open(image_path).convert("RGB")
        pil_fmt = "JPEG"
        out_media = "image/jpeg"
        quality = 85
        scale  = 1.0

        for attempt in range(8):
            buf = io.BytesIO()
            w = int(img.width  * scale)
            h = int(img.height * scale)
            resized = img.resize((max(1, w), max(1, h)), Image.LANCZOS)
            resized.save(buf, format=pil_fmt, quality=quality, optimize=True)
            if buf.tell() <= max_bytes:
                buf.seek(0)
                print(f"    [resize] {Path(image_path).name}: "
                      f"{len(raw_bytes)//1024}KB → {buf.tell()//1024}KB "
                      f"(scale={scale:.2f}, q={quality})")
                return base64.standard_b64encode(buf.read()).decode("utf-8"), out_media
            # Alternate between reducing quality and reducing dimensions
            if attempt % 2 == 0:
                quality = max(40, quality - 10)
            else:
                scale  = max(0.1, scale - 0.15)

        # Last resort: smallest possible
        buf = io.BytesIO()
        img.resize((800, 600), Image.LANCZOS).save(buf, format=pil_fmt, quality=40)
        buf.seek(0)
        return base64.standard_b64encode(buf.read()).decode("utf-8"), out_media

    except ImportError:
        # Pillow not installed — send raw and let the API error surface clearly
        print("    [warn] Pillow not installed; cannot resize large image. "
              "Run: pip install Pillow")
        return base64.standard_b64encode(raw_bytes).decode("utf-8"), media_type


def image_block(image_path: str) -> dict:
    data, media_type = encode_image(image_path)
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def call_claude(messages: list, api_key: str, system: str = SYSTEM_PROMPT,
                max_retries: int = 5) -> str:
    """
    Call Claude with exponential backoff on 429 (rate limit) errors.
    Other errors (400, 401, etc.) are raised immediately.
    """
    import time

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": messages,
    }

    wait = 15  # initial wait in seconds for 429
    for attempt in range(max_retries):
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=180,
        )
        if resp.status_code == 200:
            content = resp.json().get("content", [])
            return " ".join(b.get("text", "") for b in content
                            if b.get("type") == "text").strip()

        if resp.status_code == 429:
            if attempt < max_retries - 1:
                # Honour Retry-After header if present, else use exponential backoff
                retry_after = resp.headers.get("retry-after")
                sleep_secs  = int(retry_after) if retry_after else wait
                print(f"\n    [rate limit] 429 received. Waiting {sleep_secs}s "
                      f"(attempt {attempt+1}/{max_retries})...", end="", flush=True)
                time.sleep(sleep_secs)
                wait = min(wait * 2, 120)  # cap at 2 min
                continue
            raise RuntimeError(f"API error 429 (rate limit, {max_retries} retries exhausted): {resp.text}")

        # Non-retryable error
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")

    raise RuntimeError("call_claude: exceeded max_retries without a response")


def parse_json(text: str) -> dict:
    cleaned = re.sub(r'```(?:json)?\s*', '', text).replace('```', '').strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def find_reference_image(class_name: str, dataset_name: str):
    """
    Find a reference image from Curated_Local_Dataset/train/<dataset_name>/<class_name>/
    Falls back to any image in that folder.
    """
    class_dir = TRAIN_DIR / dataset_name / class_name
    if not class_dir.exists():
        return None
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        found = sorted(class_dir.glob(pattern))
        if found:
            return str(found[0])
    return None


# ── Interactive dataset selection ──────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def get_curated_datasets() -> list:
    """Return all category directories present in Curated_Local_Dataset/test/."""
    if not TEST_DIR.exists():
        return []
    return sorted([d.name for d in TEST_DIR.iterdir() if d.is_dir()])


def get_classes_for_dataset(dataset_name: str) -> list:
    """Return sorted list of class subdirectories for a dataset."""
    cat_dir = TEST_DIR / dataset_name
    if not cat_dir.exists():
        return []
    return sorted([d.name for d in cat_dir.iterdir() if d.is_dir()])


def count_images_in_class(dataset_name: str, class_name: str) -> int:
    cls_dir = TEST_DIR / dataset_name / class_name
    return len([f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS])


def _ask_int(prompt: str, lo: int, hi: int, zero_means_all: bool = False):
    """Keep prompting until the user enters an integer in [lo, hi] (or 0 if zero_means_all)."""
    while True:
        raw = input(prompt).strip()
        try:
            val = int(raw)
            if zero_means_all and val == 0:
                return None          # caller treats None as "all"
            if lo <= val <= hi:
                return val
            bound = f"0–{hi}" if zero_means_all else f"{lo}–{hi}"
            print(f"  Please enter a number in [{bound}].")
        except ValueError:
            print("  Please enter a valid integer.")


def _ask_indices(prompt: str, available: list, exact_count: int) -> list:
    """Ask for comma-separated 1-based indices; validate and return chosen items."""
    while True:
        raw = input(prompt).strip()
        parts = [s.strip() for s in raw.replace(" ", ",").split(",") if s.strip()]
        try:
            chosen = list(dict.fromkeys(available[int(i) - 1] for i in parts))
            if len(chosen) != exact_count:
                print(f"  Expected {exact_count} unique choice(s), got {len(chosen)}. Try again.")
                continue
            return chosen
        except (ValueError, IndexError):
            print("  Invalid input. Use the numbers shown in the list.")


def prompt_user_for_datasets() -> list:
    """
    Interactively build a run configuration.

    Returns a list of dicts, one per dataset:
      {
        "dataset":       str,
        "classes":       list[str],      # which classes to include
        "images_per_class": int | None,  # None = all available
      }
    """
    available_datasets = get_curated_datasets()
    if not available_datasets:
        print(f"ERROR: No datasets found in {TEST_DIR}")
        return []

    # ── Step 1: show datasets ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("AVAILABLE DATASETS  (Curated_Local_Dataset/test/)")
    print("=" * 60)
    for i, ds in enumerate(available_datasets, 1):
        classes   = get_classes_for_dataset(ds)
        n_images  = sum(count_images_in_class(ds, c) for c in classes)
        print(f"  [{i}] {ds}  ({len(classes)} classes, {n_images} images total)")

    # ── Step 2: how many / which datasets ─────────────────────────────────
    print()
    n_ds = _ask_int(
        f"How many datasets do you want to analyse? (1–{len(available_datasets)}): ",
        lo=1, hi=len(available_datasets)
    )

    if n_ds == len(available_datasets):
        selected_datasets = list(available_datasets)
        print(f"  → All {len(available_datasets)} datasets selected.")
    else:
        print(f"\nEnter the numbers of the {n_ds} dataset(s) (comma-separated, e.g. 1,3):")
        selected_datasets = _ask_indices("Your choice: ", available_datasets, n_ds)
        print(f"  → Selected: {selected_datasets}")

    # ── Step 3: per-dataset class & image configuration ────────────────────
    run_config = []

    for ds in selected_datasets:
        classes = get_classes_for_dataset(ds)

        print(f"\n{'─'*60}")
        print(f"  DATASET: {ds}  ({len(classes)} classes available)")
        print(f"{'─'*60}")
        for j, cls in enumerate(classes, 1):
            n = count_images_in_class(ds, cls)
            print(f"    [{j}] {cls}  ({n} test images)")

        # How many classes
        n_cls = _ask_int(
            f"\n  How many classes from '{ds}'? (1–{len(classes)}, or {len(classes)} for all): ",
            lo=1, hi=len(classes)
        )

        if n_cls == len(classes):
            selected_classes = list(classes)
            print(f"  → All {len(classes)} classes selected.")
        else:
            print(f"  Enter the numbers of the {n_cls} class(es) (comma-separated):")
            selected_classes = _ask_indices("  Your choice: ", classes, n_cls)
            print(f"  → Selected classes: {selected_classes}")

        # How many images per class
        max_imgs = max(count_images_in_class(ds, c) for c in selected_classes)
        imgs_per_class = _ask_int(
            f"  How many test images per class? (1–{max_imgs}, or 0 for all): ",
            lo=1, hi=max_imgs, zero_means_all=True
        )
        if imgs_per_class is None:
            print(f"  → All available images per class will be used.")
        else:
            print(f"  → {imgs_per_class} image(s) per class.")

        run_config.append({
            "dataset":          ds,
            "classes":          selected_classes,
            "images_per_class": imgs_per_class,
        })

    return run_config


# ── Load test images dynamically from Curated_Local_Dataset/test/ ─────────────

def load_curated_test_images(dataset_name: str,
                              selected_classes: list = None,
                              images_per_class: int = None) -> tuple:
    """
    Scan Curated_Local_Dataset/test/<dataset_name>/ to build:
      - images list: [{"name": ..., "path": ..., "ground_truth": ...}]
      - expected_classes: the *full* class list for the dataset (for prompt context)
      - active_classes:   only the classes being tested
      - description: a brief string for prompting

    selected_classes  – subset of classes to test; None = all
    images_per_class  – max images to take from each class; None = all
    """
    import random

    cat_dir = TEST_DIR / dataset_name
    if not cat_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {cat_dir}")

    all_classes = sorted([d.name for d in cat_dir.iterdir() if d.is_dir()])
    active_classes = selected_classes if selected_classes else all_classes

    images = []
    for cls in active_classes:
        cls_dir = cat_dir / cls
        if not cls_dir.exists():
            continue
        img_files = sorted(
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTS
        )
        # Deterministic shuffle so per-class limits sample fairly
        random.seed(42)
        random.shuffle(img_files)
        if images_per_class:
            img_files = img_files[:images_per_class]
        for img_file in img_files:
            images.append({
                "name":         img_file.name,
                "path":         str(img_file),
                "ground_truth": cls,
            })

    description = (
        f"{dataset_name.replace('_', ' ')} plant disease classification dataset. "
        f"All possible classes: {all_classes}"
    )
    return images, all_classes, description


# ── LLM-as-Judge ──────────────────────────────────────────────────────────────

def run_judge(
    image_path: str,
    ground_truth: str,
    prediction: str,
    confidence: float,
    reasoning_trace: list,
    api_key: str,
) -> dict:
    is_correct = prediction == ground_truth

    trace_summary = ""
    for entry in reasoning_trace:
        turn_num = entry.get("turn", "?")
        phase    = entry.get("phase", "")
        parsed   = entry.get("parsed", {})
        trace_summary += f"\n--- Turn {turn_num} ({phase}) ---\n"
        if phase == "initial_analysis":
            trace_summary += f"Prediction:     {parsed.get('prediction', 'N/A')}\n"
            trace_summary += f"Confidence:     {parsed.get('confidence', 'N/A')}\n"
            trace_summary += f"Needs ref:      {parsed.get('needs_reference', 'N/A')}\n"
            trace_summary += f"Top candidates: {parsed.get('top_candidates', [])}\n"
            trace_summary += f"Reasoning:      {parsed.get('reasoning', '')}\n"
        elif phase == "reference_comparison":
            trace_summary += f"Reference class:   {entry.get('reference_class', 'N/A')}\n"
            trace_summary += f"Updated prediction:{parsed.get('prediction', 'N/A')}\n"
            trace_summary += f"Confidence:        {parsed.get('confidence', 'N/A')}\n"
            trace_summary += f"Needs ref:         {parsed.get('needs_reference', 'N/A')}\n"
            trace_summary += f"Reasoning:         {parsed.get('reasoning', '')}\n"
        elif phase == "final_prediction":
            trace_summary += f"Prediction:    {parsed.get('prediction', 'N/A')}\n"
            trace_summary += f"Confidence:    {parsed.get('confidence', 'N/A')}\n"
            trace_summary += f"Reasoning:     {parsed.get('reasoning', parsed.get('justification', ''))}\n"

    judge_content = [
        image_block(image_path),
        text_block(f"""CONFIDENCE CALIBRATION EVALUATION

Target image is shown above.

--- OUTCOME ---
Ground truth : {ground_truth}
Prediction   : {prediction}
Result       : {"CORRECT" if is_correct else "INCORRECT"}
Stated confidence (0.0-1.0) : {confidence:.2f}

--- FULL REASONING TRACE ---
{trace_summary}

--- YOUR TASK ---
Evaluate whether the agent's stated confidence ({confidence:.2f}) was well-calibrated.
Consider:
1. The actual outcome (correct or incorrect)
2. Whether the agent needed extra reference turns (suggests lower confidence was warranted)
3. Consistency of reasoning across turns

calibration_score rubric:
  1.0 - perfectly calibrated
  0.7 - mostly calibrated, minor mismatch
  0.4 - partial mismatch
  0.1 - badly miscalibrated (overconfident when wrong, or severely underconfident when correct)

Respond ONLY with this JSON:
{{
  "outcome": "{"CORRECT" if is_correct else "INCORRECT"}",
  "stated_confidence": {confidence:.2f},
  "calibration_verdict": "<WELL_CALIBRATED | OVERCONFIDENT | UNDERCONFIDENT | INCONSISTENT>",
  "calibration_score": 0.0,
  "reasoning_consistency": "<did reasoning improve logically across turns? 1 sentence>",
  "judge_notes": "<what the agent did well or poorly in expressing confidence. 1 sentence>"
}}"""),
    ]

    try:
        resp   = call_claude([{"role": "user", "content": judge_content}], api_key, system=JUDGE_SYSTEM_PROMPT)
        parsed = parse_json(resp)
        return {"raw": resp, "parsed": parsed, "success": True}
    except Exception as e:
        return {"raw": "", "parsed": {}, "error": str(e), "success": False}


# ── Core agent loop ────────────────────────────────────────────────────────────

def classify_with_agent(
    image_path: str,
    expected_classes: list,
    dataset_description: str,
    symptoms_text: str,
    dataset_name: str,
    api_key: str,
) -> dict:
    abs_image_path    = str(Path(image_path).resolve())

    # Extract only the relevant crop/disease symptoms from the structured MD
    filtered_symptoms = extract_symptoms_for_dataset(
        symptoms_text, dataset_name, expected_classes
    )


    messages       = []
    trace          = []
    refs_viewed    = []
    ref_turns_used = 0
    turn_num       = 0

    def build_prompt() -> str:
        # Build in two parts so that filtered_symptoms (which may contain
        # literal { } characters) is concatenated, never f-string interpolated.
        header = (
            f"You are a plant disease classification agent.\n"
            f"DATASET: {dataset_description}\n"
            f"AVAILABLE CLASSES: {expected_classes}\n"
            f"TASK: Classify the disease/condition in the target image.\n"
            f"\n"
            f"KNOWLEDGE BASE (crop-wise symptom descriptions for this dataset):\n"
        )
        footer = (
            f"\nINSTRUCTIONS:\n"
            f"1. Study the KNOWLEDGE BASE above - it lists symptom descriptions for each class\n"
            f"2. Examine the target image carefully\n"
            f"3. Match the visible symptoms to the knowledge base descriptions\n"
            f"4. You MUST examine reference images for your top 2-3 candidate classes from the train directory\n"
            f"5. Make your prediction from the available classes only\n"
            f"OUTPUT: After your analysis, return a JSON object:\n"
            '{{"prediction": "class_name", "confidence": 0.0, "needs_reference": true, "top_candidates": ["class1", "class2"], "reasoning": "brief reasoning"}}\n'
            f"Rules:\n"
            f"- prediction must be exactly one of: {expected_classes}\n"
            f"- confidence: 0.0-1.0 reflecting how certain you are\n"
            f"- needs_reference: set true if confidence < 0.80 or top 2 candidates are close; set false only if clearly confident\n"
            f"- top_candidates: your ranked list of most likely classes (used to select next reference)\n"
            f"- reasoning: 1-2 sentences citing specific visual evidence matched to symptom descriptions"
        )
        return header + filtered_symptoms + footer

    def pick_reference_candidate(parsed_turn: dict, refs_viewed: list, expected_classes: list) -> str:
        candidates = parsed_turn.get("top_candidates", [])
        if candidates:
            for cls in candidates:
                if cls not in expected_classes:
                    cls = {c.lower(): c for c in expected_classes}.get(str(cls).lower(), "")
                if cls and cls not in refs_viewed:
                    return cls
        hyps = parsed_turn.get("hypotheses", [])
        if hyps:
            for h in sorted(hyps, key=lambda h: h.get("rank", 99)):
                cls = h.get("class", "")
                if cls not in expected_classes:
                    cls = {c.lower(): c for c in expected_classes}.get(cls.lower(), "")
                if cls and cls not in refs_viewed:
                    return cls
        for cls in expected_classes:
            if cls not in refs_viewed:
                return cls
        return ""

    # ── Turn 1 ────────────────────────────────────────────────────────────────
    turn_num += 1
    messages.append({"role": "user", "content": [
        image_block(image_path),
        text_block(build_prompt()),
    ]})
    try:
        resp1   = call_claude(messages, api_key)
        parsed1 = parse_json(resp1)
        messages.append({"role": "assistant", "content": resp1})
        trace.append({"turn": turn_num, "phase": "initial_analysis",
                      "raw": resp1, "parsed": parsed1})
        print(f"T{turn_num}✓", end=" ", flush=True)
    except Exception as e:
        return {"prediction": "ERROR", "trace": trace,
                "error": f"Turn {turn_num} failed: {e}", "success": False}

    needs_reference = parsed1.get("needs_reference", True)
    prediction      = parsed1.get("prediction", "")
    confidence      = float(parsed1.get("confidence", 0.5))

    # ── Reference loop ────────────────────────────────────────────────────────
    while needs_reference and ref_turns_used < MAX_REFERENCE_TURNS:
        last_parsed   = trace[-1]["parsed"] if trace else {}
        ref_candidate = pick_reference_candidate(last_parsed, refs_viewed, expected_classes)
        if not ref_candidate:
            break

        # ← KEY CHANGE: source reference from train directory
        ref_image_path = find_reference_image(ref_candidate, dataset_name)
        refs_viewed.append(ref_candidate)
        ref_turns_used += 1
        turn_num       += 1

        ref_content = [image_block(image_path)]
        if ref_image_path:
            ref_content.append(image_block(ref_image_path))
            ref_content.append(text_block(
                f'The second image above is a confirmed training example of class "{ref_candidate}". '
                f'Compare it directly with the target image (first image) before responding.'
            ))
        else:
            ref_content.append(text_block(
                f'Note: No reference image found in the train directory for class "{ref_candidate}". '
                f'Reason using knowledge base descriptions instead.'
            ))
        ref_content.append(text_block(build_prompt()))

        messages.append({"role": "user", "content": ref_content})
        try:
            resp_ref   = call_claude(messages, api_key)
            parsed_ref = parse_json(resp_ref)
            messages.append({"role": "assistant", "content": resp_ref})
            trace.append({"turn": turn_num, "phase": "reference_comparison",
                          "reference_class": ref_candidate,
                          "ref_image":       ref_image_path,
                          "raw":             resp_ref,
                          "parsed":          parsed_ref})
            print(f"T{turn_num}(ref:{ref_candidate[:8]})✓", end=" ", flush=True)
        except Exception as e:
            return {"prediction": "ERROR", "trace": trace,
                    "error": f"Reference turn {turn_num} failed: {e}", "success": False}

        needs_reference = parsed_ref.get("needs_reference", False)
        prediction      = parsed_ref.get("prediction", prediction)
        confidence      = float(parsed_ref.get("confidence", confidence))

    # ── Final turn ────────────────────────────────────────────────────────────
    turn_num += 1
    messages.append({"role": "user", "content": [
        image_block(image_path),
        text_block(
            build_prompt() +
            "\n\nThis is your FINAL turn. You MUST return a valid prediction JSON now. "
            "Set needs_reference to false. Choose your best answer from the available classes."
        ),
    ]})
    try:
        resp_final   = call_claude(messages, api_key)
        parsed_final = parse_json(resp_final)
        messages.append({"role": "assistant", "content": resp_final})
        trace.append({"turn": turn_num, "phase": "final_prediction",
                      "raw": resp_final, "parsed": parsed_final})
        print(f"T{turn_num}✓", end=" ", flush=True)
    except Exception as e:
        return {"prediction": "ERROR", "trace": trace,
                "error": f"Final turn {turn_num} failed: {e}", "success": False}

    # ── Validate ──────────────────────────────────────────────────────────────
    prediction = parsed_final.get("prediction", prediction)
    if prediction not in expected_classes:
        lower_map  = {c.lower(): c for c in expected_classes}
        prediction = lower_map.get(str(prediction).lower(), "UNKNOWN")

    confidence = float(parsed_final.get("confidence", confidence))

    reasoning_summary = (
        f"[T1] {parsed1.get('reasoning', '')}"
        + "".join(
            f"\n[T{e['turn']} ref={e.get('reference_class', '')}] "
            f"{e['parsed'].get('reasoning', '')}"
            for e in trace if e["phase"] == "reference_comparison"
        )
        + f"\n[Final] {parsed_final.get('reasoning', parsed_final.get('justification', ''))}"
    )

    return {
        "prediction":        prediction,
        "confidence":        confidence,
        "reasoning_summary": reasoning_summary,
        "ref_turns_used":    ref_turns_used,
        "refs_viewed":       refs_viewed,
        "trace":             trace,
        "num_turns":         turn_num,
        "success":           True,
    }


# ── Per-dataset runner ─────────────────────────────────────────────────────────

def run_agent_on_dataset(
    dataset_name: str,
    logs_dir: Path,
    symptoms_text: str,
    api_key: str,
    selected_classes: list = None,
    images_per_class: int = None,
) -> dict:

    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*60}")

    test_images, expected_classes, description = load_curated_test_images(
        dataset_name, selected_classes, images_per_class
    )

    print(f"Description    : {description}")
    print(f"All classes    : {expected_classes}")
    active = selected_classes if selected_classes else expected_classes
    print(f"Testing classes: {active}")
    print(f"Images/class   : {'all' if not images_per_class else images_per_class}")
    print(f"Total images   : {len(test_images)}")
    print(f"Model          : {MODEL}  |  max ref turns: {MAX_REFERENCE_TURNS}")
    print(f"Reference src  : {TRAIN_DIR / dataset_name}\n")

    dataset_logs_dir = logs_dir / dataset_name
    dataset_logs_dir.mkdir(parents=True, exist_ok=True)

    correct            = 0
    total_ref_turns    = 0
    judge_scores       = []
    calibration_counts = {"WELL_CALIBRATED": 0, "OVERCONFIDENT": 0,
                          "UNDERCONFIDENT": 0, "INCONSISTENT": 0}

    for i, img_info in enumerate(test_images):
        image_name   = img_info["name"]
        image_path   = img_info["path"]
        ground_truth = img_info["ground_truth"]

        # Small inter-image delay to reduce token-per-minute rate limit pressure
        if i > 0:
            import time
            time.sleep(5)

        print(f"  [{i+1}/{len(test_images)}] {image_name}  (gt: {ground_truth})")
        print(f"    Agent → ", end="", flush=True)

        result = classify_with_agent(
            image_path=image_path,
            expected_classes=expected_classes,
            dataset_description=description,
            symptoms_text=symptoms_text,
            dataset_name=dataset_name,
            api_key=api_key,
        )

        prediction  = result["prediction"]
        confidence  = result.get("confidence", 0.0)
        ref_turns   = result.get("ref_turns_used", 0)
        refs_viewed = result.get("refs_viewed", [])
        is_correct  = prediction == ground_truth

        if is_correct:
            correct += 1
        total_ref_turns += ref_turns

        outcome_str = "✓ CORRECT" if is_correct else f"✗ WRONG → {prediction}"
        print(f"{outcome_str}  (conf: {confidence:.2f}, ref turns: {ref_turns})")
        if refs_viewed:
            print(f"    Refs viewed → {refs_viewed}")

        final_trace = next((e for e in reversed(result["trace"]) if e["phase"] == "final_prediction"), {})
        reason = final_trace.get("parsed", {}).get("reasoning", final_trace.get("parsed", {}).get("justification", ""))
        if reason:
            print(f"    Reason → {reason[:150]}")

        print(f"    Judge  → ", end="", flush=True)
        judge_result = run_judge(
            image_path=image_path,
            ground_truth=ground_truth,
            prediction=prediction,
            confidence=confidence,
            reasoning_trace=result["trace"],
            api_key=api_key,
        )

        judge_parsed = judge_result.get("parsed", {})
        verdict      = judge_parsed.get("calibration_verdict", "UNKNOWN")
        cal_score    = float(judge_parsed.get("calibration_score", 0.0))
        judge_notes  = judge_parsed.get("judge_notes", "")

        if verdict in calibration_counts:
            calibration_counts[verdict] += 1
        judge_scores.append(cal_score)

        print(f"{verdict}  (cal score: {cal_score:.2f})")
        if judge_notes:
            print(f"           {judge_notes[:150]}")

        log_entry = {
            "image_name":        image_name,
            "ground_truth":      ground_truth,
            "prediction":        prediction,
            "correct":           is_correct,
            "confidence":        confidence,
            "ref_turns_used":    ref_turns,
            "refs_viewed":       refs_viewed,
            "reasoning_summary": result.get("reasoning_summary", ""),
            "trace":             result.get("trace", []),
            "judge": {
                "verdict":               verdict,
                "calibration_score":     cal_score,
                "reasoning_consistency": judge_parsed.get("reasoning_consistency", ""),
                "judge_notes":           judge_notes,
                "raw":                   judge_result.get("raw", ""),
            },
            "num_turns": result.get("num_turns"),
            "success":   result.get("success"),
            "error":     result.get("error"),
            "timestamp": datetime.now().isoformat(),
        }
        log_file = dataset_logs_dir / f"{Path(image_name).stem}_log.json"
        with open(log_file, "w") as f:
            json.dump(log_entry, f, indent=2)

        print()

    n        = len(test_images)
    accuracy = (correct / n) * 100 if n else 0
    avg_cal  = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0
    avg_refs = total_ref_turns / n if n else 0

    print(f"  Accuracy         : {correct}/{n} ({accuracy:.1f}%)")
    print(f"  Avg ref turns    : {avg_refs:.1f}  (max allowed: {MAX_REFERENCE_TURNS})")
    print(f"  Avg cal. score   : {avg_cal:.2f}")
    print(f"  Calibration breakdown:")
    for v, c in calibration_counts.items():
        print(f"    {v:<20}: {c}")
    print(f"  Logs             : {dataset_logs_dir}/")

    return {
        "accuracy":              accuracy,
        "avg_calibration_score": avg_cal,
        "avg_ref_turns":         avg_refs,
        "calibration_counts":    calibration_counts,
    }


# ── Main entrypoint ────────────────────────────────────────────────────────────

def run_agent(run_config=None):
    print("=" * 60)
    print("AGENTIC CLASSIFICATION (matching AGENT_FLOW.md)")
    print("  T1     : exact prompt + target image  → prediction + needs_reference?")
    print("  Loop   : if needs_reference → inject ref image from train/ → re-reason")
    print("  Final  : always runs, forces clean prediction JSON")
    print("  Judge  : calibration score immediately after prediction")
    print("=" * 60)

    if not CURATED_DIR.exists():
        print(f"\nERROR: Curated dataset not found: {CURATED_DIR}")
        print("Run 'python dataloader.py' first to create it.")
        return

    if not SYMPTOMS_FILE.exists():
        print(f"\nERROR: Symptoms file not found: {SYMPTOMS_FILE}")
        print("Run 'python generate_symptoms.py' first.")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nERROR: ANTHROPIC_API_KEY not set.")
        return

    # ── Interactive selection if not provided programmatically ──────────────
    if run_config is None:
        run_config = prompt_user_for_datasets()
        if not run_config:
            return

    symptoms_text = load_symptoms()
    print(f"\nKnowledge base   : {SYMPTOMS_FILE} ({len(symptoms_text):,} chars)")
    print(f"Reference images : {TRAIN_DIR}  (sourced from train split)")
    print(f"Max ref turns    : {MAX_REFERENCE_TURNS}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logs_dir = RESULTS_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for cfg in run_config:
        dataset_name     = cfg["dataset"]
        selected_classes = cfg.get("classes")       # None = all classes
        images_per_class = cfg.get("images_per_class")  # None = all images
        stats = run_agent_on_dataset(
            dataset_name=dataset_name,
            logs_dir=logs_dir,
            symptoms_text=symptoms_text,
            api_key=api_key,
            selected_classes=selected_classes,
            images_per_class=images_per_class,
        )
        all_results[dataset_name] = stats

    print("\n" + "=" * 60)
    print("GLOBAL SUMMARY")
    print("=" * 60)
    for dataset_name, stats in all_results.items():
        print(f"  {dataset_name}")
        print(f"    Accuracy       : {stats['accuracy']:.1f}%")
        print(f"    Avg ref turns  : {stats['avg_ref_turns']:.1f}")
        print(f"    Avg cal. score : {stats['avg_calibration_score']:.2f}")
        counts = stats["calibration_counts"]
        print(f"    Calibration    : "
              f"WELL={counts['WELL_CALIBRATED']} "
              f"OVER={counts['OVERCONFIDENT']} "
              f"UNDER={counts['UNDERCONFIDENT']} "
              f"INCON={counts['INCONSISTENT']}")

    print(f"\nFull logs: {logs_dir}/")
    return all_results


if __name__ == "__main__":
    run_agent()

    # ── Programmatic usage (bypass interactive prompts) ──────────────────────
    # run_agent(run_config=[
    #     {
    #         "dataset":          "Corn_Diseases",
    #         "classes":          ["Common_rust", "Gray_leaf_spot", "Tar_spot"],
    #         "images_per_class": 3,
    #     },
    #     {
    #         "dataset":          "Soybean_Diseases",
    #         "classes":          None,    # all classes
    #         "images_per_class": 2,
    #     },
    #     {
    #         "dataset":          "Mango_Leaf_Disease",
    #         "classes":          None,    # all classes
    #         "images_per_class": None,    # all images
    #     },
    # ])