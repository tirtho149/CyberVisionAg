#!/usr/bin/env python3
"""
Plant Disease Diagnostic Agent — Truly Agentic, KB-Driven
===========================================================
Architecture:
  • Vision Agent     — open-ended visual observation extraction (NO schema bias)
  • Diagnostic Agent — reads KB via tools, reasons freely, submits prediction
  • Judge            — calibration evaluation (disabled)

Design principles enforced throughout:
  - ZERO hardcoded disease names in any prompt, routing logic, or tool response
  - ZERO hardcoded symptom keywords or biology assumptions
  - ZERO hardcoded routing rules (no IF disease_X THEN do_Y)
  - All disease knowledge comes exclusively from disease_symptoms_crop_wise.md
  - Vision agent extracts free-form observations — no pre-baked JSON schema fields
  - Routing is done by the LLM over tool-fetched KB text, not by Python logic
  - Tools are lean data-fetchers; the LLM does all reasoning
  - Works for ANY crop / disease dataset without code changes
"""

from __future__ import annotations

import os, io, json, base64, re, time, sys, argparse, random
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import dotenv_values
import requests

# ── Environment ────────────────────────────────────────────────────────────────
env_file = Path(__file__).parent / ".env"
env_vars = dotenv_values(env_file)
for k, v in env_vars.items():
    if v:
        os.environ[k] = v

# ── Paths & constants ──────────────────────────────────────────────────────────
CURATED_DIR   = Path(__file__).parent / "Curated_Local_Dataset"
TRAIN_DIR     = CURATED_DIR / "train"
TEST_DIR      = CURATED_DIR / "test"
RESULTS_DIR   = Path(__file__).parent / "results" / "agent"
SYMPTOMS_FILE = Path(__file__).parent / "disease_symptoms_crop_wise.md"

MODEL          = "claude-sonnet-4-6"
MAX_TOKENS     = 4096
MAX_TOOL_CALLS = 16
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
MAX_LONG_EDGE  = 1024
JPEG_QUALITY   = 85

MESSAGES_URL           = "https://api.anthropic.com/v1/messages"
RETRYABLE_STATUS_CODES = {429, 502, 503, 504}


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1: VISION AGENT
# ══════════════════════════════════════════════════════════════════════════════

VISION_AGENT_PROMPT = """\
You are a visual image analyst. Your only job is to describe what you observe \
in this plant image as precisely and neutrally as possible.

You are NOT diagnosing anything.
You are NOT naming diseases or pathogens.
You are NOT using disease-specific terminology.
You are extracting raw visual observations that any careful observer would report.

Work through these four observation steps:

STEP 1 — IMAGE CONTEXT
What is shown? (e.g. single leaf, whole plant, stem close-up, seeds, roots, \
non-plant object, etc.) What plant organ(s) are visible?

STEP 2 — ABNORMALITY INVENTORY
For each distinct visual abnormality you can see, describe:
  • Location on the plant organ (edge, center, veins, surface, cross-section, etc.)
  • Approximate size (mm estimate, or relative: pinpoint / small / medium / large)
  • Shape (circular, angular, irregular, elongated, following veins, diffuse, etc.)
  • Color of the main body of the mark or lesion
  • Color of any border, margin, or surrounding halo (or state there is none)
  • Texture (sunken, raised, powdery, slimy, dry, waxy, crusty, velvety, etc.)
  • Surface structures visible (tiny dots, thread-like growth, powder, pustules, \
    galls, cysts, exudate, etc.) — describe color and size of each
  • How they are distributed across the organ (scattered, clustered, uniform, \
    confined to one area, spreading from base/tip, etc.)

STEP 3 — ABSENCE NOTES
Explicitly state 2-4 things that are NOT visible that might otherwise be expected \
for an affected plant. (e.g. "no visible powdery coating", "no yellowing around marks", \
"no raised structures on the underside", etc.)

STEP 4 — SUMMARY
In 3-5 sentences, summarise the most distinctive visual features. Which organ is \
most severely affected? What is the single most unusual or striking visual feature?

Output ONLY a JSON object with exactly these keys:
{
  "image_context": "<STEP 1 answer>",
  "abnormalities": "<STEP 2 answer>",
  "absences": "<STEP 3 answer>",
  "summary": "<STEP 4 answer>",
  "primary_organ_affected": "<leaf | stem | root | pod | seed | tassel | whole-plant | non-plant | other>",
  "approximate_growth_stage": "<seedling | young-vegetative | reproductive | late-season | unknown>"
}

CRITICAL RULES:
- Do NOT name any disease, pathogen, or scientific term implying a diagnosis
- Do NOT use terms like 'lesion', 'necrosis', 'chlorosis', 'sporulation', \
  'acervuli', 'uredinia' — describe the visual appearance instead
  (e.g. instead of 'necrosis': 'dead, dry, brown tissue'; \
   instead of 'acervuli': 'tiny black raised dots within the mark')
- Be specific about colors, sizes, and spatial relationships
- If the image is not of a plant (equipment, test strips, soil, etc.), say so clearly\
"""


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2: DIAGNOSTIC AGENT
# ══════════════════════════════════════════════════════════════════════════════

def build_diagnostic_prompt(tool_budget: int) -> str:
    return f"""\
You are a plant disease diagnostic expert. You have a set of tools that let you \
query a symptom knowledge base and retrieve reference images.

You will receive:
  1. Raw visual observations extracted from the image by a vision agent
  2. The target image itself
  3. The list of possible disease classes

Your task: reason carefully and submit exactly one prediction.

You have a budget of {tool_budget} tool calls. Use them fully — do NOT submit \
early if you have budget remaining and confidence < 0.80.

══════════════════════════════════════════
REASONING PROCESS — follow all 5 phases
══════════════════════════════════════════

PHASE 1 — ORIENT (no tools yet)
Read the observations carefully.
  • What is the primary organ affected?
  • What are the 2-3 most distinctive visual features?
  • What does the overall pattern suggest? (without naming diseases yet)

PHASE 2 — SURVEY
Call list_dataset_classes to see all available classes.
Based on the affected organ and the most striking visual features, identify \
3-5 candidate classes that could plausibly explain what was observed.
Call read_symptom_description for each candidate.

PHASE 3 — NARROW
After reading descriptions, compare each against the observations.
Which candidates clearly don't fit? Eliminate them.
For your remaining top 2-3 candidates, call get_disease_discriminators for \
EVERY pair among them (e.g. A vs B, A vs C, B vs C). Do not skip any pair.

PHASE 4 — VISUALLY CONFIRM
Call get_reference_image for your top 2-3 candidates (not just 1-2).
After viewing each, explicitly state:
  MATCH — visual features align well
  PARTIAL MATCH — some features match, some don't; explain what differs
  NO MATCH — clearly different; explain why
Adjust your ranking accordingly.

PHASE 5 — VERIFY & DECIDE
Before submitting: if confidence < 0.80 and you have budget remaining, go back \
and compare your top pick against any candidate you haven't visually confirmed yet.
Call compare_candidates to log your ruling feature and winner.
Call submit_prediction with your final answer.

══════════════════════════════════════════
TOOL USE GUIDANCE
══════════════════════════════════════════
read_symptom_description  — use broadly; read 3-6 descriptions when uncertain
get_disease_discriminators — use for EVERY uncertain pair, not just top 2
get_reference_image       — use for top 2-3 candidates; visual match matters
inspect_closely           — use when you need to re-examine a specific aspect \
                            of the target image before deciding
compare_candidates        — ALWAYS call before submit_prediction
submit_prediction         — MUST be called; never leave without a prediction

Budget strategy: you have {tool_budget} tool calls total. A typical diagnosis \
uses 3-5 for survey, 2-3 for discriminators, 2-3 for reference images, \
and 1-2 for inspect/compare/submit. If you finish early, use remaining budget \
to verify against candidates you haven't fully ruled out.

If the image appears to not be a plant (equipment, test strip, label, etc.):
  • Still call list_dataset_classes and pick the closest class
  • Submit with confidence 0.30-0.40

══════════════════════════════════════════
CONFIDENCE CALIBRATION
══════════════════════════════════════════
0.82-1.00 : Reference image nearly identical + all key features confirmed
0.68-0.81 : Strong match on reference and KB description
0.52-0.67 : Good match but some ambiguity, or image is unclear / late-stage
0.38-0.51 : Two strong candidates not fully resolved
0.25-0.37 : Best guess under significant uncertainty
Default    : 0.55 (adjust up or down based on evidence)

HARD RULE: Never submit > 0.70 if two candidates remain plausible after \
viewing reference images.\
"""


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3: JUDGE
# ══════════════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator assessing AI plant disease classification agents.
Evaluate ONLY confidence calibration: did the agent's stated confidence match \
the actual outcome?

Calibration verdicts:
  WELL_CALIBRATED  — confidence matched outcome (high+correct or low+wrong)
  OVERCONFIDENT    — stated high confidence (>0.7) but was incorrect
  UNDERCONFIDENT   — stated low confidence (<0.4) but was correct
  INCONSISTENT     — confidence did not reflect the quality of reasoning shown

Respond ONLY with valid JSON:\
"""


# ══════════════════════════════════════════════════════════════════════════════
#  TOOLS
# ══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "list_dataset_classes",
        "description": (
            "Return the complete list of disease class names available in this dataset. "
            "Always call this first so you know all possible predictions."
        ),
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "read_symptom_description",
        "description": (
            "Fetch the visual symptom description for a specific disease class "
            "from the knowledge base. Returns the full text description exactly as "
            "written in the KB — no processing or summarisation. "
            "Call this for multiple candidates to compare against your observations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "disease_name": {
                    "type": "string",
                    "description": "Exact class name as listed in list_dataset_classes"
                }
            },
            "required": ["disease_name"]
        }
    },
    {
        "name": "get_disease_discriminators",
        "description": (
            "Given two candidate disease class names, returns the key visual features "
            "that distinguish them — derived dynamically from their KB descriptions. "
            "Use this whenever you are uncertain between two candidates. "
            "The result is generated fresh from the KB; no pre-set rules are applied."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "disease_a": {
                    "type": "string",
                    "description": "First candidate class name"
                },
                "disease_b": {
                    "type": "string",
                    "description": "Second candidate class name"
                }
            },
            "required": ["disease_a", "disease_b"]
        }
    },
    {
        "name": "get_reference_image",
        "description": (
            "Fetch a confirmed training example image for a disease class. "
            "Compare it visually to the target image and state "
            "MATCH / PARTIAL MATCH / NO MATCH with specific visual reasons. "
            "Use for your top 1-2 candidates to confirm or refute them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "class_name": {
                    "type": "string",
                    "description": "Exact class name"
                }
            },
            "required": ["class_name"]
        }
    },
    {
        "name": "inspect_closely",
        "description": (
            "Re-examine the TARGET IMAGE with focused attention on a specific visual "
            "aspect you need to verify before deciding. Use this when you are uncertain "
            "about a particular feature (e.g. texture of a mark, color of a border, "
            "presence of surface structures). State what you examined and what you found."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "aspect_to_check": {
                    "type": "string",
                    "description": "The specific visual feature you want to verify"
                },
                "your_finding": {
                    "type": "string",
                    "description": "Your careful assessment of that feature after re-examining"
                }
            },
            "required": ["aspect_to_check", "your_finding"]
        }
    },
    {
        "name": "compare_candidates",
        "description": (
            "Log your final comparison between your top two candidates. "
            "Identify the single visual feature that decided between them. "
            "ALWAYS call this before submit_prediction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_a":    {"type": "string"},
                "candidate_b":    {"type": "string"},
                "evidence_for_a": {"type": "string", "description": "Visual evidence supporting A"},
                "evidence_for_b": {"type": "string", "description": "Visual evidence supporting B"},
                "ruling_feature": {
                    "type": "string",
                    "description": "The single visual feature that decided between the two"
                },
                "winner": {
                    "type": "string",
                    "description": "Winning class name and one-sentence justification"
                }
            },
            "required": ["candidate_a", "candidate_b", "evidence_for_a",
                         "evidence_for_b", "ruling_feature", "winner"]
        }
    },
    {
        "name": "submit_prediction",
        "description": (
            "Submit your final disease classification. "
            "Call compare_candidates first. "
            "This ends the diagnostic loop."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prediction": {
                    "type": "string",
                    "description": (
                        "Your predicted class. Must exactly match one of the class names "
                        "returned by list_dataset_classes."
                    )
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence 0.0-1.0. Default 0.55 unless evidence clearly supports higher or lower."
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "2-4 sentences explaining: (1) the visual features that were most "
                        "decisive, (2) the closest alternative and why it was ruled out."
                    )
                }
            },
            "required": ["prediction", "confidence", "reasoning"]
        }
    }
]


# ══════════════════════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════

def load_symptoms() -> str:
    with open(SYMPTOMS_FILE, "r") as f:
        return f.read()


def parse_symptoms_by_crop(symptoms_text: str) -> dict:
    parsed = {}
    current_crop, current_disease, collecting, buf = "", "", False, []

    for line in symptoms_text.splitlines():
        crop_m = re.match(r"^##\s+Crop:\s+(.+)$", line)
        if crop_m:
            if current_crop and current_disease and buf:
                parsed.setdefault(current_crop, {})[current_disease] = " ".join(buf).strip()
            current_crop = crop_m.group(1).strip()
            current_disease, buf, collecting = "", [], False
            parsed.setdefault(current_crop, {})
            continue

        dis_m = re.match(r"^###\s+(.+)$", line)
        if dis_m:
            if current_crop and current_disease and buf:
                parsed.setdefault(current_crop, {})[current_disease] = " ".join(buf).strip()
            current_disease = dis_m.group(1).strip().replace(" ", "_")
            buf, collecting = [], False
            continue

        if line.strip() == "**Symptoms:**":
            collecting, buf = True, []
            continue
        if line.strip() == "**Reference Images:**":
            collecting = False
            continue
        if collecting and line.strip() and not line.startswith("-"):
            buf.append(line.strip())

    if current_crop and current_disease and buf:
        parsed.setdefault(current_crop, {})[current_disease] = " ".join(buf).strip()

    return parsed


REGISTRY_OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "disease_registry" / "outputs"


def load_symptoms_from_xlsx(source: str, crop: str, dataset_name: str) -> str:
    """Load symptoms from disease_registry xlsx and convert to markdown format."""
    import openpyxl

    xlsx_path = REGISTRY_OUTPUTS_DIR / f"{crop}_{source}.xlsx"
    if not xlsx_path.exists():
        print(f"\nERROR: xlsx not found: {xlsx_path}")
        sys.exit(1)

    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active

    lines = [f"## Crop: {dataset_name.replace('_', ' ')}\n"]
    count = 0
    for row in ws.iter_rows(min_row=2, values_only=True):
        disease_name = row[0]  # Col A
        visual_desc = row[4]   # Col E (Visual Description)
        if not disease_name:
            continue
        underscored = disease_name.strip().replace(" ", "_")
        lines.append(f"### {underscored}\n")
        lines.append("**Symptoms:**")
        lines.append(visual_desc.strip() if visual_desc else "(no description available)")
        lines.append("")
        count += 1

    wb.close()
    print(f"\nKB source: {xlsx_path.name} ({count} diseases loaded)")
    return "\n".join(lines)


def build_symptom_lookup(symptoms_text: str, dataset_name: str) -> dict:
    all_syms = parse_symptoms_by_crop(symptoms_text)
    norm = dataset_name.replace("_", " ").lower()
    matched = next((k for k in all_syms if k.lower() == norm), None)
    if not matched:
        matched = next((k for k in all_syms
                        if norm in k.lower() or k.lower() in norm), None)
    return all_syms.get(matched, {})


# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMIC DISCRIMINATOR
# ══════════════════════════════════════════════════════════════════════════════

_DISCRIMINATOR_CACHE: dict = {}

_DISCRIMINATOR_SYSTEM = """\
You are a plant pathology expert. You will be given the knowledge-base symptom \
descriptions for two diseases. Your job is to identify the single most important \
visual feature that distinguishes them from each other.

Base your answer ONLY on the descriptions provided — do not add knowledge from elsewhere.

Respond with valid JSON only (no markdown, no preamble):
{
  "ruling_feature": "<the single most discriminating visual trait — 1 concise line>",
  "disease_a_markers": "<the key visual markers of disease A in 1-2 sentences>",
  "disease_b_markers": "<the key visual markers of disease B in 1-2 sentences>",
  "decision_rule": "<an IF/THEN decision rule a field agent can apply — 1-2 sentences>"
}\
"""


def get_discriminators(disease_a: str, disease_b: str,
                       symptom_lookup: dict, api_key: str) -> str:
    cache_key = tuple(sorted([disease_a, disease_b]))
    if cache_key in _DISCRIMINATOR_CACHE:
        return _DISCRIMINATOR_CACHE[cache_key]

    def _get_sym(name: str) -> str:
        norm = name.replace(" ", "_")
        sym = symptom_lookup.get(norm)
        if not sym:
            lc = {k.lower(): v for k, v in symptom_lookup.items()}
            sym = lc.get(norm.lower())
        return sym or f"(No knowledge-base entry found for '{name}')"

    sym_a = _get_sym(disease_a)
    sym_b = _get_sym(disease_b)

    prompt = (
        f"Disease A: {disease_a}\n"
        f"Knowledge-base description:\n{sym_a}\n\n"
        f"Disease B: {disease_b}\n"
        f"Knowledge-base description:\n{sym_b}\n\n"
        "Based ONLY on these descriptions, identify the single most important "
        "visual feature that distinguishes A from B."
    )

    try:
        resp = call_claude_api(
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            system=_DISCRIMINATOR_SYSTEM,
            max_retries=4,
        )
        text = extract_text(resp.get("content", []))
        parsed = parse_json(text)

        if parsed and "ruling_feature" in parsed:
            result = (
                f"DISCRIMINATING: '{disease_a}' vs '{disease_b}'\n"
                f"Ruling feature : {parsed['ruling_feature']}\n"
                f"{disease_a}: {parsed.get('disease_a_markers', sym_a[:300])}\n"
                f"{disease_b}: {parsed.get('disease_b_markers', sym_b[:300])}\n"
                f"Decision rule  : {parsed.get('decision_rule', 'Compare the markers above.')}"
            )
        else:
            result = (
                f"DISCRIMINATING: '{disease_a}' vs '{disease_b}'\n"
                f"(Raw KB descriptions — compare manually)\n"
                f"{disease_a}: {sym_a[:500]}\n"
                f"{disease_b}: {sym_b[:500]}"
            )

    except Exception as e:
        result = (
            f"Discriminator generation failed for '{disease_a}' vs '{disease_b}': {e}\n"
            f"{disease_a}: {sym_a[:300]}\n"
            f"{disease_b}: {sym_b[:300]}"
        )

    _DISCRIMINATOR_CACHE[cache_key] = result
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def compress_image_to_bytes(image_path: str) -> tuple:
    try:
        from PIL import Image as PILImage
        img = PILImage.open(image_path).convert("RGB")
        w, h = img.size
        long_edge = max(w, h)
        if long_edge > MAX_LONG_EDGE:
            scale = MAX_LONG_EDGE / long_edge
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                             PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        buf.seek(0)
        return buf.read(), "image/jpeg"
    except ImportError:
        raw = Path(image_path).read_bytes()
        ext = Path(image_path).suffix.lower()
        mt = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
              ".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")
        return raw, mt


def inline_image_block(image_path: str) -> dict:
    img_bytes, mime_type = compress_image_to_bytes(image_path)
    data = base64.standard_b64encode(img_bytes).decode("utf-8")
    return {"type": "image", "source": {"type": "base64",
                                         "media_type": mime_type, "data": data}}


# ══════════════════════════════════════════════════════════════════════════════
#  MESSAGES API
# ══════════════════════════════════════════════════════════════════════════════

def _messages_headers(api_key: str) -> dict:
    return {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


def call_claude_api(messages: list, api_key: str, system: str,
                    tools: list = None, max_retries: int = 8) -> dict:
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools

    wait = 15
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                MESSAGES_URL,
                headers=_messages_headers(api_key),
                json=payload,
                timeout=180,
            )
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                time.sleep(wait)
                wait = min(wait * 2, 120)
                continue
            raise RuntimeError(f"Connection error: {e}")

        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
            sleep = int(resp.headers.get("retry-after", wait))
            time.sleep(sleep)
            wait = min(wait * 2, 120)
            continue
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")

    raise RuntimeError("call_claude_api: max retries exceeded")


def extract_text(content_blocks: list) -> str:
    return " ".join(
        b.get("text", "") for b in content_blocks if b.get("type") == "text"
    ).strip()


def parse_json(text: str) -> dict:
    cleaned = re.sub(r'```(?:json)?\s*', '', text).replace('```', '').strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{[\s\S]*\}', cleaned)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


# ══════════════════════════════════════════════════════════════════════════════
#  REFERENCE IMAGE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def find_reference_image(class_name: str, dataset_name: str):
    class_dir = TRAIN_DIR / dataset_name / class_name
    if not class_dir.exists():
        return None
    all_images = []
    for pat in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        all_images.extend(class_dir.glob(pat))
    if not all_images:
        return None
    all_images = sorted(all_images)
    return str(all_images[len(all_images) // 2])


# ══════════════════════════════════════════════════════════════════════════════
#  VISION AGENT — Stage 1
# ══════════════════════════════════════════════════════════════════════════════

def run_vision_agent(image_path: str, api_key: str) -> dict:
    img_block = inline_image_block(image_path)

    messages = [{
        "role": "user",
        "content": [
            img_block,
            {"type": "text",
             "text": "Describe what you observe in this image, following your instructions. "
                     "Return ONLY the JSON object."}
        ]
    }]

    try:
        resp = call_claude_api(
            messages, api_key,
            system=VISION_AGENT_PROMPT,
            )
        text = extract_text(resp.get("content", []))
        features = parse_json(text)
        if features:
            organ = features.get("primary_organ_affected", "?")
            stage = features.get("approximate_growth_stage", "?")
            print(f"[vision:organ={organ} stage={stage}]", end=" ", flush=True)
        return features
    except Exception as e:
        print(f"[vision-err:{e}]", end=" ", flush=True)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL EXECUTOR
# ══════════════════════════════════════════════════════════════════════════════

def execute_tool(tool_name: str, tool_input: dict,
                 dataset_name: str, expected_classes: list,
                 symptom_lookup: dict, api_key: str,
                 ref_budget_remaining: int | None = None) -> tuple:

    if tool_name == "list_dataset_classes":
        text = (
            f"Available classes in '{dataset_name}' "
            f"({len(expected_classes)} total):\n"
            + "\n".join(f"  {i+1:2d}. {c}"
                        for i, c in enumerate(sorted(expected_classes)))
        )
        return [{"type": "text", "text": text}], {"tool": tool_name}

    if tool_name == "read_symptom_description":
        disease = tool_input.get("disease_name", "")
        key = disease.replace(" ", "_")
        symptom = symptom_lookup.get(key)
        if not symptom:
            lc = {k.lower(): v for k, v in symptom_lookup.items()}
            symptom = lc.get(key.lower())

        if symptom:
            text = (
                f"Knowledge-base entry for '{disease}':\n"
                f"{'─' * 40}\n"
                f"{symptom}\n"
                f"{'─' * 40}"
            )
        else:
            available = sorted(symptom_lookup.keys())
            text = (
                f"No KB entry found for '{disease}'.\n"
                f"Available entries: {available[:15]}"
                + (" (+ more)" if len(available) > 15 else "")
            )
        return [{"type": "text", "text": text}], {"tool": tool_name, "disease": disease}

    if tool_name == "get_disease_discriminators":
        a = tool_input.get("disease_a", "")
        b = tool_input.get("disease_b", "")
        text = get_discriminators(a, b, symptom_lookup, api_key)
        return [{"type": "text", "text": text}], {
            "tool": tool_name, "disease_a": a, "disease_b": b
        }

    if tool_name == "get_reference_image":
        cls = tool_input.get("class_name", "")
        if ref_budget_remaining is not None and ref_budget_remaining <= 0:
            return (
                [{"type": "text",
                  "text": f"Reference image budget exhausted (--k limit reached). "
                          "Use read_symptom_description or submit_prediction instead."}],
                {"tool": tool_name, "class": cls, "ref_image": None,
                 "budget_exhausted": True}
            )
        ref_path = find_reference_image(cls, dataset_name)
        if not ref_path:
            return (
                [{"type": "text",
                  "text": f"No reference image found for '{cls}'. "
                          "Try read_symptom_description for its description instead."}],
                {"tool": tool_name, "class": cls, "ref_image": None}
            )
        img_block = inline_image_block(ref_path)
        blocks = [
            img_block,
            {"type": "text",
             "text": (f"Reference training image for '{cls}'.\n"
                      "Compare this to the target image and state: "
                      "MATCH / PARTIAL MATCH / NO MATCH — with specific visual reasons.")}
        ]
        return blocks, {
            "tool": tool_name, "class": cls,
            "ref_image": ref_path
        }

    if tool_name == "inspect_closely":
        aspect  = tool_input.get("aspect_to_check", "")
        finding = tool_input.get("your_finding", "")
        text = (
            f"Focused inspection logged.\n"
            f"Aspect examined : {aspect}\n"
            f"Finding         : {finding}\n"
            "Use this finding to refine your candidate ranking."
        )
        return [{"type": "text", "text": text}], {
            "tool": tool_name, "aspect": aspect, "finding": finding
        }

    if tool_name == "compare_candidates":
        a      = tool_input.get("candidate_a", "")
        b      = tool_input.get("candidate_b", "")
        ruling = tool_input.get("ruling_feature", "")
        winner = tool_input.get("winner", "")
        text = (
            f"Comparison logged: {a} vs {b}\n"
            f"Evidence for {a}: {tool_input.get('evidence_for_a', '')}\n"
            f"Evidence for {b}: {tool_input.get('evidence_for_b', '')}\n"
            f"Ruling feature  : {ruling}\n"
            f"Winner          : {winner}"
        )
        return [{"type": "text", "text": text}], {
            "tool": tool_name,
            "candidate_a": a, "candidate_b": b,
            "ruling_feature": ruling, "winner": winner
        }

    if tool_name == "submit_prediction":
        prediction = tool_input.get("prediction", "UNKNOWN")
        confidence = float(tool_input.get("confidence", 0.0))
        reasoning  = tool_input.get("reasoning", "")
        return (
            [{"type": "text",
              "text": f"Prediction submitted: {prediction} (confidence={confidence:.2f})"}],
            {"tool": tool_name, "prediction": prediction,
             "confidence": confidence, "reasoning": reasoning}
        )

    return [{"type": "text", "text": f"Unknown tool: {tool_name}"}], {"tool": tool_name}


# ══════════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC AGENT LOOP — Stage 2
# ══════════════════════════════════════════════════════════════════════════════

def _error_result(trace: list, error_msg: str,
                  tool_calls: int, refs_viewed: list) -> dict:
    return {
        "prediction": "UNKNOWN", "confidence": 0.0, "reasoning": "",
        "trace": trace, "tool_call_count": tool_calls,
        "refs_viewed": refs_viewed, "symptoms_read": [],
        "comparisons": [], "success": False, "error": error_msg,
        "num_turns": len(trace), "reasoning_summary": "",
        "vision_features": {},
    }


def classify_with_agent(image_path: str, expected_classes: list,
                        dataset_description: str, symptoms_text: str,
                        dataset_name: str, api_key: str,
                        max_ref_views: int | None = None) -> dict:

    symptom_lookup = build_symptom_lookup(symptoms_text, dataset_name)
    diagnostic_prompt = build_diagnostic_prompt(MAX_TOOL_CALLS)

    test_img_block = inline_image_block(image_path)

    vision_features = run_vision_agent(image_path, api_key)
    observations_str = json.dumps(vision_features, indent=2) if vision_features else "unavailable"

    initial_prompt = (
        "══════════════════════════════════════\n"
        "VISUAL OBSERVATIONS (from vision agent)\n"
        "══════════════════════════════════════\n"
        f"{observations_str}\n\n"
        "══════════════════════════════════════\n"
        "YOUR TASK\n"
        "══════════════════════════════════════\n"
        "The target image is attached. Use your tools and your REASONING PROCESS "
        "(Phases 1-5) to classify this plant disease image.\n\n"
        f"Dataset context: {dataset_description}\n\n"
        "Start by calling list_dataset_classes to see all possible predictions, "
        "then follow Phases 1-5 from your system prompt.\n\n"
        "You MUST call submit_prediction with exactly one class from the available list."
    )

    messages = [{
        "role": "user",
        "content": [test_img_block, {"type": "text", "text": initial_prompt}]
    }]

    trace: list       = []
    tool_call_count   = 0
    ref_view_count    = 0
    prediction        = "UNKNOWN"
    confidence        = 0.0
    reasoning         = ""
    refs_viewed: list = []
    symptoms_read: list = []
    comparisons: list = []
    submitted         = False

    while tool_call_count < MAX_TOOL_CALLS and not submitted:
        try:
            resp = call_claude_api(
                messages, api_key,
                system=diagnostic_prompt,
                tools=TOOLS,
            )
        except Exception as e:
            print(f"\n    [API ERROR] {e}", flush=True)
            return _error_result(trace, str(e), tool_call_count, refs_viewed)

        stop_reason    = resp.get("stop_reason", "")
        content_blocks = resp.get("content", [])
        messages.append({"role": "assistant", "content": content_blocks})
        tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]

        if not tool_uses:
            assistant_text = extract_text(content_blocks)
            parsed = parse_json(assistant_text)
            if parsed.get("prediction"):
                prediction = parsed["prediction"]
                confidence = float(parsed.get("confidence", 0.5))
                reasoning  = parsed.get("reasoning", assistant_text[:300])
            trace.append({"phase": "no_tool", "raw": assistant_text})
            print(f"[stop:{stop_reason}]", end=" ", flush=True)
            break

        tool_result_blocks = []
        for tu in tool_uses:
            tool_name  = tu.get("name", "")
            tool_input = tu.get("input", {})
            tool_id    = tu.get("id", "")
            tool_call_count += 1

            ref_budget = (max_ref_views - ref_view_count
                         if max_ref_views is not None else None)
            result_blocks, meta = execute_tool(
                tool_name, tool_input,
                dataset_name, expected_classes,
                symptom_lookup, api_key,
                ref_budget_remaining=ref_budget,
            )

            trace.append({
                "phase": "tool_call",
                "tool": tool_name,
                "input": tool_input,
                "tool_id": tool_id,
                "meta": meta,
                "result_text": extract_text(result_blocks),
            })

            if tool_name == "get_reference_image":
                cls = tool_input.get("class_name", "")
                if not meta.get("budget_exhausted"):
                    ref_view_count += 1
                if cls and cls not in refs_viewed:
                    refs_viewed.append(cls)
                print(f"[ref:{cls[:10]}]", end=" ", flush=True)
            elif tool_name == "read_symptom_description":
                dis = tool_input.get("disease_name", "")
                if dis and dis not in symptoms_read:
                    symptoms_read.append(dis)
                print(f"[sym:{dis[:10]}]", end=" ", flush=True)
            elif tool_name == "get_disease_discriminators":
                a = tool_input.get("disease_a", "")
                b = tool_input.get("disease_b", "")
                print(f"[disc:{a[:7]}v{b[:7]}]", end=" ", flush=True)
            elif tool_name == "inspect_closely":
                print("[inspect]", end=" ", flush=True)
            elif tool_name == "compare_candidates":
                comparisons.append(meta.get("winner", ""))
                print(f"[cmp:{meta.get('ruling_feature','')[:15]}]", end=" ", flush=True)
            elif tool_name == "list_dataset_classes":
                print("[list]", end=" ", flush=True)
            elif tool_name == "submit_prediction":
                prediction = meta["prediction"]
                confidence = meta["confidence"]
                reasoning  = meta["reasoning"]
                submitted  = True
                trace.append({
                    "phase": "final_prediction",
                    "parsed": {
                        "prediction": prediction,
                        "confidence": confidence,
                        "reasoning":  reasoning,
                    },
                    "raw": "submit_prediction called",
                })
                print("[submit]", end=" ", flush=True)

            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": result_blocks,
            })

        messages.append({"role": "user", "content": tool_result_blocks})
        if submitted:
            break

    if prediction == "UNKNOWN":
        print("[force-final]", end=" ", flush=True)
        force_msg = (
            f"You have used {tool_call_count} tool calls and must now submit immediately. "
            "Call submit_prediction with your best guess. "
            f"Classes: {sorted(expected_classes)[:10]}"
            + (" (+ more)" if len(expected_classes) > 10 else "")
        )
        messages.append({"role": "user", "content": force_msg})
        try:
            resp = call_claude_api(
                messages, api_key,
                system=diagnostic_prompt,
                tools=TOOLS,
            )
            for b in resp.get("content", []):
                if b.get("type") == "tool_use" and b.get("name") == "submit_prediction":
                    inp = b.get("input", {})
                    prediction = inp.get("prediction", "UNKNOWN")
                    confidence = float(inp.get("confidence", 0.25))
                    reasoning  = inp.get("reasoning", "forced final prediction")
                    trace.append({
                        "phase": "final_prediction",
                        "parsed": {"prediction": prediction,
                                   "confidence": confidence,
                                   "reasoning": reasoning},
                        "raw": "forced submit",
                    })
                    break
            if prediction == "UNKNOWN":
                text = extract_text(resp.get("content", []))
                parsed = parse_json(text)
                prediction = parsed.get("prediction",
                                        sorted(expected_classes)[0] if expected_classes
                                        else "UNKNOWN")
                confidence = float(parsed.get("confidence", 0.25))
                reasoning  = parsed.get("reasoning", "forced fallback")
        except Exception as e:
            prediction = sorted(expected_classes)[0] if expected_classes else "UNKNOWN"
            confidence = 0.25
            reasoning  = f"forced fallback after error: {e}"

    return {
        "prediction":       prediction,
        "confidence":       confidence,
        "reasoning":        reasoning,
        "trace":            trace,
        "tool_call_count":  tool_call_count,
        "refs_viewed":      refs_viewed,
        "symptoms_read":    symptoms_read,
        "comparisons":      comparisons,
        "success":          prediction != "UNKNOWN",
        "error":            None,
        "num_turns":        len(trace),
        "reasoning_summary": reasoning[:200] if reasoning else "",
        "vision_features":  vision_features,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  JUDGE — Stage 3
# ══════════════════════════════════════════════════════════════════════════════

def run_judge(image_path: str, ground_truth: str, prediction: str,
              confidence: float, reasoning_trace: list, api_key: str) -> dict:

    trace_summary = []
    for entry in reasoning_trace[-8:]:
        tool   = entry.get("tool", entry.get("phase", ""))
        result = entry.get("result_text", "")[:120]
        trace_summary.append(f"  [{tool}]: {result}")

    judge_prompt = (
        f"Ground truth : {ground_truth}\n"
        f"Prediction   : {prediction}\n"
        f"Confidence   : {confidence:.2f}\n"
        f"Correct      : {prediction == ground_truth}\n\n"
        "Agent reasoning trace (last steps):\n"
        + "\n".join(trace_summary)
        + "\n\n"
        "Evaluate calibration. Respond with JSON only:\n"
        "{\n"
        '  "calibration_verdict": "<WELL_CALIBRATED|OVERCONFIDENT|UNDERCONFIDENT|INCONSISTENT>",\n'
        '  "calibration_score": <0.0 to 1.0>,\n'
        '  "reasoning_consistency": "<brief note>",\n'
        '  "judge_notes": "<1-2 sentences on agent performance>"\n'
        "}"
    )
    try:
        resp = call_claude_api(
            messages=[{"role": "user", "content": judge_prompt}],
            api_key=api_key,
            system=JUDGE_SYSTEM_PROMPT,
            max_retries=4,
        )
        text = extract_text(resp.get("content", []))
        parsed = parse_json(text)
        return {"parsed": parsed, "raw": text}
    except Exception as e:
        return {"parsed": {
            "calibration_verdict": "UNKNOWN",
            "calibration_score": 0.0,
            "reasoning_consistency": "",
            "judge_notes": str(e),
        }, "raw": ""}


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def get_expected_classes(dataset_name: str, selected_classes=None) -> list:
    for base in (TEST_DIR, TRAIN_DIR):
        dataset_dir = base / dataset_name
        if dataset_dir.exists():
            all_classes = [d.name for d in dataset_dir.iterdir()
                           if d.is_dir() and not d.name.startswith(".")]
            if selected_classes:
                all_classes = [c for c in all_classes if c in selected_classes]
            return sorted(all_classes)
    return []


def get_test_images(dataset_name: str, expected_classes: list,
                    images_per_class=None) -> list:
    test_images = []
    for cls in expected_classes:
        cls_dir = TEST_DIR / dataset_name / cls
        if not cls_dir.exists():
            continue
        images = sorted(f for f in cls_dir.iterdir()
                        if f.suffix.lower() in IMAGE_EXTS)
        if images_per_class:
            images = images[:images_per_class]
        for img in images:
            test_images.append((str(img), cls, img.name))
    return test_images


def _safe_input(prompt: str) -> str:
    """Read a line from stdin, stripping whitespace. Returns '' on EOF/interrupt."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def _parse_number_list(raw: str, lo: int, hi: int):
    """
    Parse a comma-separated string of integers, each in [lo, hi].
    Returns sorted unique list, or None on any parse/range error.
    """
    try:
        nums = [int(x.strip()) for x in raw.split(",") if x.strip()]
        if not nums:
            return None
        if any(n < lo or n > hi for n in nums):
            return None
        return sorted(set(nums))
    except ValueError:
        return None


def _count_images(ds_name: str, cls_name: str) -> int:
    """Count test images for a given dataset + class."""
    cls_dir = TEST_DIR / ds_name / cls_name
    if not cls_dir.exists():
        return 0
    return sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def prompt_user_for_datasets() -> list:
    """
    Interactive dataset/class selector matching the expected UI:

    ============================================================
    AVAILABLE DATASETS  (Curated_Local_Dataset/test/)
    ============================================================
      [1] Alfalfa_Diseases  (1 classes, 10 images total)
      ...
    How many datasets? (1-N): 1
    Enter the numbers of the 1 dataset(s) (comma-separated):
    Your choice: 4
      -> Selected: ['Soybean_Diseases']
    ------------------------------------------------------------
      DATASET: Soybean_Diseases  (32 classes available)
    ------------------------------------------------------------
        [1] Anthracnose  (10 test images)
        ...
      How many classes from 'Soybean_Diseases'? (1-32, or 32 for all): 1
      Enter the numbers of the 1 class(es) (comma-separated):
      Your choice: 11
        -> Selected classes: ['Frogeye_leaf_spot']
      Images per class for 'Soybean_Diseases'? (Enter for all, or a number): 3
        -> 3 image(s) per class
    """
    base = TEST_DIR if TEST_DIR.exists() else (TRAIN_DIR if TRAIN_DIR.exists() else None)
    if base is None:
        print("ERROR: Neither test nor train directory found.")
        return []

    # ── Discover datasets ────────────────────────────────────────────────────
    all_datasets = sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if not all_datasets:
        print(f"No datasets found in {base}")
        return []

    # Pre-compute per-dataset class list and total image count
    ds_meta: dict = {}
    for ds in all_datasets:
        classes = sorted(
            d.name for d in (base / ds).iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        total_imgs = sum(_count_images(ds, c) for c in classes)
        ds_meta[ds] = {"classes": classes, "total_images": total_imgs}

    # ── Print dataset menu ───────────────────────────────────────────────────
    print("=" * 60)
    print(f"AVAILABLE DATASETS  ({base.relative_to(Path(__file__).parent)}/)")
    print("=" * 60)
    for i, ds in enumerate(all_datasets, 1):
        m = ds_meta[ds]
        print(f"  [{i}] {ds}  ({len(m['classes'])} classes, {m['total_images']} images total)")

    n_ds = len(all_datasets)

    # ── How many datasets? ───────────────────────────────────────────────────
    while True:
        raw = _safe_input(f"How many datasets? (1-{n_ds}): ")
        if not raw:
            return []
        try:
            want = int(raw)
            if 1 <= want <= n_ds:
                break
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {n_ds}.")

    # ── Which datasets? ──────────────────────────────────────────────────────
    selected_ds_names: list = []
    while True:
        print(f"Enter the numbers of the {want} dataset(s) (comma-separated):")
        raw = _safe_input("Your choice: ")
        if not raw:
            return []
        nums = _parse_number_list(raw, 1, n_ds)
        if nums is None or len(nums) != want:
            print(f"  Please enter exactly {want} unique number(s) between 1 and {n_ds}.")
            continue
        selected_ds_names = [all_datasets[n - 1] for n in nums]
        print(f"  -> Selected: {selected_ds_names}")
        break

    # ── Per-dataset: class selection + images-per-class ──────────────────────
    run_config = []
    for ds in selected_ds_names:
        classes = ds_meta[ds]["classes"]
        n_cls   = len(classes)

        print("-" * 60)
        print(f"  DATASET: {ds}  ({n_cls} classes available)")
        print("-" * 60)
        for i, cls in enumerate(classes, 1):
            n_img = _count_images(ds, cls)
            print(f"    [{i}] {cls}  ({n_img} test images)")

        # How many classes?
        while True:
            raw = _safe_input(
                f"  How many classes from '{ds}'? (1-{n_cls}, or {n_cls} for all): "
            )
            if not raw:
                return []
            try:
                want_cls = int(raw)
                if 1 <= want_cls <= n_cls:
                    break
            except ValueError:
                pass
            print(f"  Please enter a number between 1 and {n_cls}.")

        # Which classes?
        if want_cls == n_cls:
            selected_classes = None   # all
            print(f"    -> All {n_cls} classes selected.")
        else:
            selected_classes = None
            while True:
                print(f"  Enter the numbers of the {want_cls} class(es) (comma-separated):")
                raw = _safe_input("  Your choice: ")
                if not raw:
                    return []
                nums = _parse_number_list(raw, 1, n_cls)
                if nums is None or len(nums) != want_cls:
                    print(f"  Please enter exactly {want_cls} unique number(s) between 1 and {n_cls}.")
                    continue
                selected_classes = [classes[n - 1] for n in nums]
                print(f"    -> Selected classes: {selected_classes}")
                break

        # Images per class
        while True:
            raw = _safe_input(
                f"  Images per class for '{ds}'? (press Enter for all, or enter a number): "
            )
            if raw == "":
                ipc = None
                print(f"    -> All images per class.")
                break
            try:
                ipc = int(raw)
                if ipc > 0:
                    print(f"    -> {ipc} image(s) per class.")
                    break
            except ValueError:
                pass
            print("  Please enter a positive integer or press Enter for all.")

        run_config.append({
            "dataset":          ds,
            "classes":          selected_classes,
            "images_per_class": ipc,
        })

    return run_config


def _process_single_image(idx, total, image_path, ground_truth, image_name,
                          expected_classes, dataset_description, symptoms_text,
                          dataset_name, api_key, max_ref_views, dataset_logs_dir):
    """Classify one image. Returns a result dict. Thread-safe."""
    print(f"\n[{idx}/{total}] {image_name} | GT: {ground_truth}")
    print("  ", end="", flush=True)

    result = classify_with_agent(
        image_path=image_path,
        expected_classes=expected_classes,
        dataset_description=dataset_description,
        symptoms_text=symptoms_text,
        dataset_name=dataset_name,
        api_key=api_key,
        max_ref_views=max_ref_views,
    )

    prediction   = result.get("prediction", "UNKNOWN")
    confidence   = result.get("confidence", 0.0)
    tool_calls   = result.get("tool_call_count", 0)
    refs_viewed  = result.get("refs_viewed", [])
    syms_read    = result.get("symptoms_read", [])
    comparisons  = result.get("comparisons", [])
    is_correct   = (prediction == ground_truth)

    print(f"\n  GT={ground_truth} | Pred={prediction} | "
          f"{'✓' if is_correct else '✗'} "
          f"(conf:{confidence:.2f} tools:{tool_calls})")
    if refs_viewed:
        print(f"    Refs viewed  -> {refs_viewed}")
    if syms_read:
        print(f"    Syms read    -> {syms_read}")
    if comparisons:
        print(f"    Comparisons  -> {comparisons}")

    vf = result.get("vision_features", {})
    if vf:
        print(f"    Vision       -> "
              f"organ={vf.get('primary_organ_affected','?')} "
              f"stage={vf.get('approximate_growth_stage','?')}")

    final_entry = next(
        (e for e in reversed(result["trace"])
         if e.get("phase") == "final_prediction"), {}
    )
    reason = final_entry.get("parsed", {}).get("reasoning", "")
    if reason:
        print(f"    Reason       -> {reason[:200]}")

    log_entry = {
        "image_name":       image_name,
        "ground_truth":     ground_truth,
        "prediction":       prediction,
        "correct":          is_correct,
        "confidence":       confidence,
        "tool_call_count":  tool_calls,
        "refs_viewed":      refs_viewed,
        "symptoms_read":    syms_read,
        "comparisons_made": comparisons,
        "vision_features":  vf,
        "reasoning_summary": result.get("reasoning_summary", ""),
        "trace":            result.get("trace", []),
        "num_turns":  result.get("num_turns"),
        "success":    result.get("success"),
        "error":      result.get("error"),
        "timestamp":  datetime.now().isoformat(),
    }
    log_file = dataset_logs_dir / f"{Path(image_name).stem}_log.json"
    with open(log_file, "w") as f:
        json.dump(log_entry, f, indent=2)
    print()

    return {
        "is_correct": is_correct,
        "tool_calls": tool_calls,
        "refs_viewed": refs_viewed,
    }


def run_agent_on_dataset(dataset_name: str, logs_dir: Path,
                         symptoms_text: str, api_key: str,
                         selected_classes=None, images_per_class=None,
                         max_ref_views: int | None = None,
                         parallel: int = 1) -> dict:

    expected_classes = get_expected_classes(dataset_name, selected_classes)
    if not expected_classes:
        print(f"  No classes found for '{dataset_name}'")
        return {}

    test_images = get_test_images(dataset_name, expected_classes, images_per_class)
    if not test_images:
        print(f"  No test images found for '{dataset_name}'")
        return {}

    dataset_description = (
        f"Dataset '{dataset_name}' — {len(expected_classes)} classes, "
        f"{len(test_images)} test images total"
    )

    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_name}")
    print(f"Classes : {len(expected_classes)}")
    print(f"Images  : {len(test_images)}")
    if parallel > 1:
        print(f"Parallel: {parallel} workers")
    print(f"{'='*60}")

    dataset_logs_dir = logs_dir / dataset_name
    dataset_logs_dir.mkdir(parents=True, exist_ok=True)

    correct     = 0
    total_refs  = 0
    total_tools = 0

    def process(idx_and_image):
        idx, (image_path, ground_truth, image_name) = idx_and_image
        return _process_single_image(
            idx, len(test_images), image_path, ground_truth, image_name,
            expected_classes, dataset_description, symptoms_text,
            dataset_name, api_key, max_ref_views, dataset_logs_dir,
        )

    if parallel <= 1:
        image_results = [process((idx, img))
                         for idx, img in enumerate(test_images, 1)]
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(process, (idx, img)): idx
                for idx, img in enumerate(test_images, 1)
            }
            image_results = []
            for future in as_completed(futures):
                image_results.append(future.result())

    for r in image_results:
        if r["is_correct"]:
            correct += 1
        total_refs  += len(r["refs_viewed"])
        total_tools += r["tool_calls"]

    n         = len(test_images)
    accuracy  = (correct / n) * 100 if n else 0
    avg_refs  = total_refs  / n if n else 0
    avg_tools = total_tools / n if n else 0

    print(f"\n  Accuracy         : {correct}/{n} ({accuracy:.1f}%)")
    print(f"  Avg tool calls   : {avg_tools:.1f}  (ceiling: {MAX_TOOL_CALLS})")
    print(f"  Avg ref images   : {avg_refs:.1f}")
    print(f"  Logs             : {dataset_logs_dir}/")

    return {
        "accuracy":              accuracy,
        "avg_ref_turns":         avg_refs,
        "avg_tool_calls":        avg_tools,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_agent(run_config=None, symptom_source="default", max_ref_views=None, parallel=1):
    print("=" * 60)
    print("AGENTIC CLASSIFICATION  (tool-use + inline base64)")
    print(f"  Model         : {MODEL}")
    print(f"  Max tool calls: {MAX_TOOL_CALLS} per image")
    print(f"  Ref image cap : {max_ref_views if max_ref_views is not None else 'unlimited'}")
    print(f"  Symptom source: {symptom_source}")
    print(f"  Image size    : max {MAX_LONG_EDGE}px long edge, JPEG q{JPEG_QUALITY}")
    print(f"  Image format  : inline base64 (JPEG q{JPEG_QUALITY}, max {MAX_LONG_EDGE}px)")
    print("=" * 60)

    if not CURATED_DIR.exists():
        print(f"\nERROR: Curated dataset not found: {CURATED_DIR}")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nERROR: ANTHROPIC_API_KEY not set.")
        return

    if run_config is None:
        run_config = prompt_user_for_datasets()
        if not run_config:
            print("No datasets selected. Exiting.")
            return

    # Load KB based on symptom source
    dataset_name = run_config[0]["dataset"]
    if symptom_source == "default":
        if not SYMPTOMS_FILE.exists():
            print(f"\nERROR: Symptoms file not found: {SYMPTOMS_FILE}")
            return
        symptoms_text = load_symptoms()
        print(f"\nKnowledge base   : {SYMPTOMS_FILE.name} ({len(symptoms_text):,} chars)")
    else:
        crop = dataset_name.replace("_Diseases", "").replace("_Disease", "")
        symptoms_text = load_symptoms_from_xlsx(symptom_source, crop, dataset_name)
        print(f"  KB size: {len(symptoms_text):,} chars")

    # Print KB coverage against dataset classes
    lookup = build_symptom_lookup(symptoms_text, dataset_name)
    expected = get_expected_classes(dataset_name)
    if expected:
        matched = [c for c in expected if lookup.get(c) or lookup.get(c.replace(" ", "_"))]
        missing = [c for c in expected if c not in matched]
        print(f"  KB coverage: {len(matched)}/{len(expected)} diseases have symptom text")
        if missing:
            print(f"  Missing KB entries: {', '.join(missing[:10])}"
                  + (f" (+{len(missing)-10} more)" if len(missing) > 10 else ""))

    # Results dir separated by source
    results_base = Path(__file__).parent / "results" / "agent" / symptom_source
    results_base.mkdir(parents=True, exist_ok=True)
    logs_dir = results_base / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict = {}
    for cfg in run_config:
        stats = run_agent_on_dataset(
            dataset_name=cfg["dataset"],
            logs_dir=logs_dir,
            symptoms_text=symptoms_text,
            api_key=api_key,
            selected_classes=cfg.get("classes"),
            images_per_class=cfg.get("images_per_class"),
            max_ref_views=max_ref_views,
            parallel=parallel,
        )
        all_results[cfg["dataset"]] = stats

    print("\n" + "=" * 60)
    print("GLOBAL SUMMARY")
    print("=" * 60)
    for ds, stats in all_results.items():
        if not stats:
            continue
        print(f"  {ds}")
        print(f"    Accuracy       : {stats['accuracy']:.1f}%")
        print(f"    Avg tool calls : {stats['avg_tool_calls']:.1f}")
        print(f"    Avg ref images : {stats['avg_ref_turns']:.1f}")

    print(f"\nFull logs: {logs_dir}/")
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Agentic plant disease classification")
    parser.add_argument("--symptom-source", default="default",
                        choices=["default", "local", "internet"],
                        help="KB source: default (markdown), local (PDF xlsx), internet (web xlsx)")
    parser.add_argument("--dataset", default="Soybean_Diseases",
                        help="Dataset folder name under Curated_Local_Dataset/")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Limit to N random classes")
    parser.add_argument("--images-per-class", type=int, default=None,
                        help="Test images per class")
    parser.add_argument("--quick-test", type=int, default=None, metavar="N",
                        help="Shortcut: sets num-classes=N, images-per-class=1")
    parser.add_argument("--k", type=int, default=None,
                        help="Max get_reference_image calls per image")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of images to process concurrently (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for class selection (default: 42)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # --quick-test is a shortcut
    num_classes = args.num_classes
    images_per_class = args.images_per_class
    if args.quick_test is not None:
        num_classes = args.quick_test
        images_per_class = 1

    # Build run_config from CLI args
    random.seed(args.seed)
    selected_classes = None
    if num_classes is not None:
        all_classes = get_expected_classes(args.dataset)
        if all_classes and num_classes < len(all_classes):
            selected_classes = sorted(random.sample(all_classes, num_classes))
        else:
            selected_classes = all_classes

    run_config = [{
        "dataset": args.dataset,
        "classes": selected_classes,
        "images_per_class": images_per_class,
    }]

    run_agent(
        run_config=run_config,
        symptom_source=args.symptom_source,
        max_ref_views=args.k,
        parallel=args.parallel,
    )