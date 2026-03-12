#!/usr/bin/env python3
"""
Plant Disease Diagnostic Agent — Truly Agentic, KB-Driven
===========================================================
Architecture:
  • Vision Agent     — open-ended visual observation extraction (NO schema bias)
  • Diagnostic Agent — reads KB via tools, reasons freely, submits prediction
  • Judge            — calibration evaluation

Design principles enforced throughout:
  - ZERO hardcoded disease names in any prompt, routing logic, or tool response
  - ZERO hardcoded symptom keywords or biology assumptions
  - ZERO hardcoded routing rules (no IF disease_X THEN do_Y)
  - All disease knowledge comes exclusively from crop_disease_registry_updated.xlsx
  - Vision agent extracts free-form observations — no pre-baked JSON schema fields
  - Routing is done by the LLM over tool-fetched KB text, not by Python logic
  - Tools are lean data-fetchers; the LLM does all reasoning
  - Works for ANY crop / disease dataset without code changes

Sync with generate_symptoms.py
────────────────────────────────
  KB source  : crop_disease_registry_updated.xlsx
  Sheet 2    : Local_Registry  — Crop, Disease, Pathogen, Type, Affected Parts, Visual Description, Confidence
  Sheet 3    : Registry  — same + Transmission, Look-alikes, Economic Importance, Favorable Conditions, Conflicts, Citations
  Ref images : Curated_Dataset/Reference_Image/<Crop>/<Disease>/
  Test images: Curated_Dataset/Benchmark/<Crop>/<Disease>/

Ablation modes (--ablation flag):
  local  — Local_Registry only (doc extraction, faster, no web cost)
  web    — Registry only (web-search enriched, superset of local — default)
"""

import os, io, json, base64, re, time, sys
from pathlib import Path
from datetime import datetime
from dotenv import dotenv_values
import requests

# ── Environment ────────────────────────────────────────────────────────────────
env_file = Path(__file__).parent / ".env"
env_vars = dotenv_values(env_file)
for k, v in env_vars.items():
    if v:
        os.environ[k] = v

# ── Paths & constants ──────────────────────────────────────────────────────────
CURATED_DIR   = Path(__file__).parent / "Curated_Dataset"
REF_IMAGE_DIR = CURATED_DIR / "Reference_Image"
BENCH_DIR     = CURATED_DIR / "Benchmark"
RESULTS_DIR   = Path(__file__).parent / "results" / "agent"
REGISTRY_PATH = Path(__file__).parent / "crop_disease_registry_updated.xlsx"

MODEL          = "claude-sonnet-4-6"
MAX_TOKENS     = 4096
MAX_TOOL_CALLS = 16
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
MAX_LONG_EDGE  = 1024
JPEG_QUALITY   = 85

# ── Ablation configuration ─────────────────────────────────────────────────────
VALID_ABLATION_MODES = {"local", "web"}
DEFAULT_ABLATION     = "web"

# xlsx sheet / column names (must match generate_symptoms.py exactly)
SUMMARY_SHEET  = "Summary"
LOCAL_REGISTRY_SHEET = "Local_Registry"
REGISTRY_SHEET = "Registry"
COL_CROP      = "Crop"
COL_DISEASE   = "Disease"
COL_VISUAL    = "Visual Description"   # column name in both registry sheets

FILES_API_BETA         = "files-api-2025-04-14"
FILES_BASE_URL         = "https://api.anthropic.com/v1/files"
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

DIAGNOSTIC_AGENT_PROMPT = """\
You are a plant disease diagnostic expert. You have a set of tools that let you \
query a symptom knowledge base and retrieve reference images.

You will receive:
  1. Raw visual observations extracted from the image by a vision agent
  2. The target image itself
  3. The list of possible disease classes
  4. The active KB ablation mode — which symptom source(s) are available

Your task: reason carefully and submit exactly one prediction.

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
For your remaining top 2 candidates, call get_disease_discriminators to find \
the single feature that separates them.

PHASE 4 — VISUALLY CONFIRM
Call get_reference_image for your top 1-2 candidates.
After viewing each, explicitly state:
  MATCH — visual features align well
  PARTIAL MATCH — some features match, some don't; explain what differs
  NO MATCH — clearly different; explain why
Adjust your confidence accordingly.

PHASE 5 — DECIDE
Call compare_candidates to log your ruling feature and winner.
Call submit_prediction with your final answer.

══════════════════════════════════════════
TOOL USE GUIDANCE
══════════════════════════════════════════
read_symptom_description   — use broadly; read 3-6 descriptions when uncertain
get_disease_discriminators — use for every pair where you are uncertain
get_reference_image        — use for your top 1-2 candidates; visual match matters
inspect_closely            — use when you need to re-examine a specific aspect \
                             of the target image before deciding
compare_candidates         — ALWAYS call before submit_prediction
submit_prediction          — MUST be called; never leave without a prediction

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
                "disease_a": {"type": "string", "description": "First candidate class name"},
                "disease_b": {"type": "string", "description": "Second candidate class name"}
            },
            "required": ["disease_a", "disease_b"]
        }
    },
    {
        "name": "get_reference_image",
        "description": (
            "Fetch a confirmed reference image for a disease class. "
            "Compare it visually to the target image and state "
            "MATCH / PARTIAL MATCH / NO MATCH with specific visual reasons. "
            "Use for your top 1-2 candidates to confirm or refute them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "class_name": {"type": "string", "description": "Exact class name"}
            },
            "required": ["class_name"]
        }
    },
    {
        "name": "inspect_closely",
        "description": (
            "Re-examine the TARGET IMAGE with focused attention on a specific visual "
            "aspect you need to verify before deciding."
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
                    "description": "Confidence 0.0-1.0. Default 0.55."
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "2-4 sentences: (1) most decisive visual features, "
                        "(2) closest alternative and why it was ruled out."
                    )
                }
            },
            "required": ["prediction", "confidence", "reasoning"]
        }
    }
]


# ══════════════════════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE — reads directly from xlsx (no .md dependency)
# ══════════════════════════════════════════════════════════════════════════════

def load_kb_from_xlsx(registry_path: Path, ablation: str = DEFAULT_ABLATION) -> dict:
    """
    Read crop_disease_registry_updated.xlsx and build a flat symptom lookup:
        { "Disease_Name_underscored": "symptom text …" }

    Reads Visual Description from:
      A  → Local_Registry sheet  (local doc extraction — faster, no web cost)
      B  → Registry sheet  (web-search enriched — superset of A)

    Also pulls Pathogen, Type, Affected Parts as metadata prefix.
    """
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl required: pip install openpyxl")

    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    sheet_name = REGISTRY_SHEET if ablation == "web" else LOCAL_REGISTRY_SHEET

    wb = openpyxl.load_workbook(registry_path, read_only=True, data_only=True)
    if sheet_name not in wb.sheetnames:
        raise ValueError(f"Sheet '{sheet_name}' not found in {registry_path.name}. "
                         f"Run generate_symptoms.py first.")

    ws   = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return {}

    header = [str(c).strip() if c else "" for c in rows[0]]

    def _ci(name):
        try:   return header.index(name)
        except ValueError: return -1

    ci_disease = _ci(COL_DISEASE)
    ci_visual  = _ci(COL_VISUAL)
    ci_path    = _ci("Pathogen")
    ci_parts   = _ci("Affected Parts")

    lookup: dict = {}
    for row in rows[1:]:
        if not row or all(c is None for c in row):
            continue
        disease = str(row[ci_disease]).strip() if ci_disease >= 0 and ci_disease < len(row) and row[ci_disease] else ""
        if not disease or disease.startswith("="):
            continue
        visual   = str(row[ci_visual]).strip()  if ci_visual  >= 0 and ci_visual  < len(row) and row[ci_visual]  else ""
        pathogen = str(row[ci_path]).strip()    if ci_path    >= 0 and ci_path    < len(row) and row[ci_path]    else ""
        parts    = str(row[ci_parts]).strip()   if ci_parts   >= 0 and ci_parts   < len(row) and row[ci_parts]   else ""

        text_parts = []
        if pathogen: text_parts.append(f"Pathogen: {pathogen}")
        if parts:    text_parts.append(f"Affected parts: {parts}")
        if visual:   text_parts.append(visual)

        key = disease.replace(" ", "_")
        lookup[key] = "\n".join(text_parts)

    return lookup


def get_symptom(disease_name: str, symptom_lookup: dict) -> str:
    """Look up disease with exact → case-insensitive → suffix fallback."""
    key = disease_name.replace(" ", "_")
    if key in symptom_lookup:
        return symptom_lookup[key]
    lc  = {k.lower(): v for k, v in symptom_lookup.items()}
    val = lc.get(key.lower(), "")
    if val:
        return val
    for k_lc, v in lc.items():
        if k_lc.endswith(key.lower()):
            return v
    return ""


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

    sym_a = get_symptom(disease_a, symptom_lookup) or f"(No KB entry for '{disease_a}')"
    sym_b = get_symptom(disease_b, symptom_lookup) or f"(No KB entry for '{disease_b}')"

    prompt = (
        f"Disease A: {disease_a}\nKB description:\n{sym_a}\n\n"
        f"Disease B: {disease_b}\nKB description:\n{sym_b}\n\n"
        "Identify the single most important visual feature that distinguishes A from B."
    )
    try:
        resp   = call_claude_api(
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key, system=_DISCRIMINATOR_SYSTEM,
            max_retries=4, use_files_beta=False,
        )
        parsed = parse_json(extract_text(resp.get("content", [])))
        if parsed and "ruling_feature" in parsed:
            result = (
                f"DISCRIMINATING: '{disease_a}' vs '{disease_b}'\n"
                f"Ruling feature : {parsed['ruling_feature']}\n"
                f"{disease_a}: {parsed.get('disease_a_markers', sym_a[:300])}\n"
                f"{disease_b}: {parsed.get('disease_b_markers', sym_b[:300])}\n"
                f"Decision rule  : {parsed.get('decision_rule', 'Compare markers above.')}"
            )
        else:
            result = (
                f"DISCRIMINATING: '{disease_a}' vs '{disease_b}'\n"
                f"{disease_a}: {sym_a[:500]}\n{disease_b}: {sym_b[:500]}"
            )
    except Exception as e:
        result = (
            f"Discriminator failed: {e}\n"
            f"{disease_a}: {sym_a[:300]}\n{disease_b}: {sym_b[:300]}"
        )
    _DISCRIMINATOR_CACHE[cache_key] = result
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def compress_image_to_bytes(image_path: str) -> tuple:
    try:
        from PIL import Image as PILImage
        img       = PILImage.open(image_path).convert("RGB")
        w, h      = img.size
        long_edge = max(w, h)
        if long_edge > MAX_LONG_EDGE:
            scale = MAX_LONG_EDGE / long_edge
            img   = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                               PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        buf.seek(0)
        return buf.read(), "image/jpeg"
    except ImportError:
        raw = Path(image_path).read_bytes()
        ext = Path(image_path).suffix.lower()
        mt  = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".png": "image/png",  ".webp": "image/webp"}.get(ext, "image/jpeg")
        return raw, mt


def file_image_block(file_id: str) -> dict:
    return {"type": "image", "source": {"type": "file", "file_id": file_id}}


def inline_image_block(image_path: str) -> dict:
    img_bytes, mime_type = compress_image_to_bytes(image_path)
    data = base64.standard_b64encode(img_bytes).decode("utf-8")
    return {"type": "image", "source": {"type": "base64",
                                         "media_type": mime_type, "data": data}}


# ══════════════════════════════════════════════════════════════════════════════
#  FILES API
# ══════════════════════════════════════════════════════════════════════════════

def _files_headers(api_key: str) -> dict:
    return {"x-api-key": api_key, "anthropic-version": "2023-06-01",
            "anthropic-beta": FILES_API_BETA}


def upload_image(image_path: str, api_key: str, max_retries: int = 8):
    img_bytes, mime_type = compress_image_to_bytes(image_path)
    filename = Path(image_path).stem + ".jpg"
    wait = 10
    for attempt in range(max_retries):
        try:
            resp = requests.post(FILES_BASE_URL, headers=_files_headers(api_key),
                                 files={"file": (filename, img_bytes, mime_type)},
                                 timeout=60)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout):
            if attempt < max_retries - 1:
                time.sleep(wait); wait = min(wait * 2, 60); continue
            return None
        if resp.status_code == 200:
            return resp.json().get("id") or None
        if resp.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
            time.sleep(int(resp.headers.get("retry-after", wait)))
            wait = min(wait * 2, 60); continue
        return None
    return None


def delete_file(file_id: str, api_key: str):
    try:
        requests.delete(f"{FILES_BASE_URL}/{file_id}",
                        headers=_files_headers(api_key), timeout=15)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  MESSAGES API
# ══════════════════════════════════════════════════════════════════════════════

def _messages_headers(api_key: str, include_files_beta: bool = False) -> dict:
    h = {"x-api-key": api_key, "anthropic-version": "2023-06-01",
         "content-type": "application/json"}
    if include_files_beta:
        h["anthropic-beta"] = FILES_API_BETA
    return h


def call_claude_api(messages: list, api_key: str, system: str,
                    tools: list = None, max_retries: int = 8,
                    use_files_beta: bool = False) -> dict:
    payload = {"model": MODEL, "max_tokens": MAX_TOKENS,
               "system": system, "messages": messages}
    if tools:
        payload["tools"] = tools
    wait = 15
    for attempt in range(max_retries):
        try:
            resp = requests.post(MESSAGES_URL,
                                 headers=_messages_headers(api_key, use_files_beta),
                                 json=payload, timeout=180)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                time.sleep(wait); wait = min(wait * 2, 120); continue
            raise RuntimeError(f"Connection error: {e}")
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
            time.sleep(int(resp.headers.get("retry-after", wait)))
            wait = min(wait * 2, 120); continue
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
    raise RuntimeError("call_claude_api: max retries exceeded")


def extract_text(content_blocks: list) -> str:
    return " ".join(b.get("text", "") for b in content_blocks
                    if b.get("type") == "text").strip()


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
#  Curated_Dataset/Reference_Image/<Crop>/<Disease>/
# ══════════════════════════════════════════════════════════════════════════════

def _name_variants(name: str) -> list:
    return list(dict.fromkeys([
        name, name.replace(" ", "_"),
        name.lower(), name.lower().replace(" ", "_"),
    ]))


def find_reference_image(class_name: str, crop_name: str = ""):
    search_roots = []
    if crop_name:
        for v in _name_variants(crop_name):
            c = REF_IMAGE_DIR / v
            if c.is_dir():
                search_roots.append(c); break
    if not search_roots and REF_IMAGE_DIR.exists():
        search_roots = [d for d in REF_IMAGE_DIR.iterdir() if d.is_dir()]
    for crop_dir in search_roots:
        for v in _name_variants(class_name):
            disease_dir = crop_dir / v
            if disease_dir.is_dir():
                images = sorted(f for f in disease_dir.iterdir()
                                if f.suffix.lower() in IMAGE_EXTS)
                if images:
                    return str(images[len(images) // 2])
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK IMAGE DISCOVERY
#  Curated_Dataset/Benchmark/<Crop>/<Disease>/
# ══════════════════════════════════════════════════════════════════════════════

def get_crop_class_map() -> dict:
    """Returns { crop_name: [disease1, disease2, ...] } from Benchmark/"""
    result: dict = {}
    if not BENCH_DIR.exists():
        return result
    for crop_dir in sorted(BENCH_DIR.iterdir()):
        if not crop_dir.is_dir() or crop_dir.name.startswith("."):
            continue
        diseases = sorted(d.name for d in crop_dir.iterdir()
                          if d.is_dir() and not d.name.startswith("."))
        if diseases:
            result[crop_dir.name] = diseases
    return result


def get_bench_images(crop_name: str, class_name: str,
                     images_per_class: int = None) -> list:
    crop_dir = None
    for v in _name_variants(crop_name):
        c = BENCH_DIR / v
        if c.is_dir():
            crop_dir = c; break
    if not crop_dir:
        return []
    disease_dir = None
    for v in _name_variants(class_name):
        d = crop_dir / v
        if d.is_dir():
            disease_dir = d; break
    if not disease_dir:
        return []
    images = sorted(str(f) for f in disease_dir.iterdir()
                    if f.suffix.lower() in IMAGE_EXTS)
    if images_per_class:
        images = images[:images_per_class]
    return [(p, class_name, Path(p).name) for p in images]


def count_bench_images(crop_name: str, class_name: str) -> int:
    return len(get_bench_images(crop_name, class_name))


# ══════════════════════════════════════════════════════════════════════════════
#  VISION AGENT — Stage 1
# ══════════════════════════════════════════════════════════════════════════════

def run_vision_agent(image_path: str, test_file_id, api_key: str) -> dict:
    img_block = (file_image_block(test_file_id) if test_file_id
                 else inline_image_block(image_path))
    messages  = [{"role": "user", "content": [
        img_block,
        {"type": "text",
         "text": "Describe what you observe in this image, following your instructions. "
                 "Return ONLY the JSON object."}
    ]}]
    try:
        resp     = call_claude_api(messages, api_key, system=VISION_AGENT_PROMPT,
                                   use_files_beta=(test_file_id is not None))
        features = parse_json(extract_text(resp.get("content", [])))
        if features:
            print(f"[vision:organ={features.get('primary_organ_affected','?')} "
                  f"stage={features.get('approximate_growth_stage','?')}]",
                  end=" ", flush=True)
        return features
    except Exception as e:
        print(f"[vision-err:{e}]", end=" ", flush=True)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL EXECUTOR
# ══════════════════════════════════════════════════════════════════════════════

def execute_tool(tool_name: str, tool_input: dict,
                 crop_name: str, expected_classes: list,
                 symptom_lookup: dict, ref_file_cache: dict,
                 api_key: str) -> tuple:

    if tool_name == "list_dataset_classes":
        text = (
            f"Available classes ({len(expected_classes)} total):\n"
            + "\n".join(f"  {i+1:2d}. {c}"
                        for i, c in enumerate(sorted(expected_classes)))
        )
        return [{"type": "text", "text": text}], {"tool": tool_name}

    if tool_name == "read_symptom_description":
        disease = tool_input.get("disease_name", "")
        symptom = get_symptom(disease, symptom_lookup)
        if symptom:
            text = (f"Knowledge-base entry for '{disease}':\n"
                    f"{'─'*40}\n{symptom}\n{'─'*40}")
        else:
            available = sorted(symptom_lookup.keys())
            text = (f"No KB entry found for '{disease}'.\n"
                    f"Available (sample): {available[:15]}"
                    + (" (+ more)" if len(available) > 15 else ""))
        return [{"type": "text", "text": text}], {"tool": tool_name, "disease": disease}

    if tool_name == "get_disease_discriminators":
        a, b = tool_input.get("disease_a", ""), tool_input.get("disease_b", "")
        text  = get_discriminators(a, b, symptom_lookup, api_key)
        return [{"type": "text", "text": text}], {
            "tool": tool_name, "disease_a": a, "disease_b": b}

    if tool_name == "get_reference_image":
        cls      = tool_input.get("class_name", "")
        ref_path = find_reference_image(cls, crop_name)
        if not ref_path:
            return (
                [{"type": "text",
                  "text": f"No reference image found for '{cls}'. "
                          "Try read_symptom_description instead."}],
                {"tool": tool_name, "class": cls, "ref_image": None, "file_id": None}
            )
        if cls not in ref_file_cache:
            ref_file_cache[cls] = upload_image(ref_path, api_key)
        file_id   = ref_file_cache[cls]
        img_block = (file_image_block(file_id) if file_id
                     else inline_image_block(ref_path))
        return (
            [img_block,
             {"type": "text",
              "text": (f"Reference image for '{cls}'.\n"
                       "Compare to target: MATCH / PARTIAL MATCH / NO MATCH "
                       "— with specific visual reasons.")}],
            {"tool": tool_name, "class": cls,
             "ref_image": ref_path, "file_id": file_id}
        )

    if tool_name == "inspect_closely":
        aspect  = tool_input.get("aspect_to_check", "")
        finding = tool_input.get("your_finding", "")
        text = (f"Focused inspection logged.\n"
                f"Aspect : {aspect}\nFinding: {finding}\n"
                "Use this to refine your candidate ranking.")
        return [{"type": "text", "text": text}], {
            "tool": tool_name, "aspect": aspect, "finding": finding}

    if tool_name == "compare_candidates":
        a, b   = tool_input.get("candidate_a", ""), tool_input.get("candidate_b", "")
        ruling = tool_input.get("ruling_feature", "")
        winner = tool_input.get("winner", "")
        text = (f"Comparison logged: {a} vs {b}\n"
                f"Evidence for {a}: {tool_input.get('evidence_for_a','')}\n"
                f"Evidence for {b}: {tool_input.get('evidence_for_b','')}\n"
                f"Ruling feature  : {ruling}\nWinner          : {winner}")
        return [{"type": "text", "text": text}], {
            "tool": tool_name,
            "candidate_a": a, "candidate_b": b,
            "ruling_feature": ruling, "winner": winner}

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

def _error_result(trace, error_msg, tool_calls, refs_viewed) -> dict:
    return {
        "prediction": "UNKNOWN", "confidence": 0.0, "reasoning": "",
        "trace": trace, "tool_call_count": tool_calls,
        "refs_viewed": refs_viewed, "symptoms_read": [],
        "comparisons": [], "success": False, "error": error_msg,
        "num_turns": len(trace), "reasoning_summary": "",
        "vision_features": {}, "ablation": "",
    }


def classify_with_agent(image_path: str, expected_classes: list,
                        dataset_description: str, symptom_lookup: dict,
                        crop_name: str, ablation: str,
                        api_key: str, ref_file_cache: dict) -> dict:

    test_file_id   = upload_image(image_path, api_key)
    print(f"[{'file-api' if test_file_id else 'inline-b64'}]", end=" ", flush=True)
    test_img_block = (file_image_block(test_file_id) if test_file_id
                      else inline_image_block(image_path))

    vision_features  = run_vision_agent(image_path, test_file_id, api_key)
    observations_str = json.dumps(vision_features, indent=2) if vision_features else "unavailable"

    active_sources = []
    if ablation == "local": active_sources.append("📄 local=Local_Registry (doc extraction)")
    if ablation == "web":   active_sources.append("🌐 web=Registry (web-search enriched)")

    initial_prompt = (
        "══════════════════════════════════════\n"
        "VISUAL OBSERVATIONS (from vision agent)\n"
        "══════════════════════════════════════\n"
        f"{observations_str}\n\n"
        "══════════════════════════════════════\n"
        f"ACTIVE KB SOURCE  : {ablation}\n"
        "══════════════════════════════════════\n"
        + "\n".join(f"  {s}" for s in active_sources) + "\n\n"
        "══════════════════════════════════════\n"
        "YOUR TASK\n"
        "══════════════════════════════════════\n"
        "The target image is attached. Follow REASONING PROCESS Phases 1-5.\n\n"
        f"Dataset context: {dataset_description}\n\n"
        "Start by calling list_dataset_classes, then follow Phases 1-5.\n"
        "You MUST call submit_prediction with exactly one class from the list."
    )

    messages = [{"role": "user", "content": [
        test_img_block, {"type": "text", "text": initial_prompt}
    ]}]

    trace: list         = []
    tool_call_count     = 0
    prediction          = "UNKNOWN"
    confidence          = 0.0
    reasoning           = ""
    refs_viewed: list   = []
    symptoms_read: list = []
    comparisons: list   = []
    submitted           = False

    while tool_call_count < MAX_TOOL_CALLS and not submitted:
        try:
            resp = call_claude_api(
                messages, api_key,
                system=DIAGNOSTIC_AGENT_PROMPT, tools=TOOLS,
                use_files_beta=(test_file_id is not None),
            )
        except Exception as e:
            print(f"\n    [API ERROR] {e}", flush=True)
            if test_file_id: delete_file(test_file_id, api_key)
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
            tool_name       = tu.get("name", "")
            tool_input      = tu.get("input", {})
            tool_id         = tu.get("id", "")
            tool_call_count += 1

            result_blocks, meta = execute_tool(
                tool_name, tool_input,
                crop_name, expected_classes,
                symptom_lookup, ref_file_cache, api_key,
            )

            trace.append({
                "phase": "tool_call", "tool": tool_name,
                "input": tool_input, "tool_id": tool_id,
                "meta": meta, "result_text": extract_text(result_blocks),
            })

            if tool_name == "get_reference_image":
                cls = tool_input.get("class_name", "")
                if cls and cls not in refs_viewed: refs_viewed.append(cls)
                print(f"[ref:{cls[:10]}]", end=" ", flush=True)
            elif tool_name == "read_symptom_description":
                dis = tool_input.get("disease_name", "")
                if dis and dis not in symptoms_read: symptoms_read.append(dis)
                print(f"[sym:{dis[:10]}]", end=" ", flush=True)
            elif tool_name == "get_disease_discriminators":
                a = tool_input.get("disease_a", ""); b = tool_input.get("disease_b", "")
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
                    "parsed": {"prediction": prediction,
                               "confidence": confidence, "reasoning": reasoning},
                    "raw": "submit_prediction called",
                })
                print("[submit]", end=" ", flush=True)

            tool_result_blocks.append({
                "type": "tool_result", "tool_use_id": tool_id,
                "content": result_blocks,
            })

        messages.append({"role": "user", "content": tool_result_blocks})
        if submitted:
            break

    if prediction == "UNKNOWN":
        print("[force-final]", end=" ", flush=True)
        force_msg = (
            f"You have used {tool_call_count} tool calls — submit immediately. "
            "Call submit_prediction now. "
            f"Classes: {sorted(expected_classes)[:10]}"
            + (" (+ more)" if len(expected_classes) > 10 else "")
        )
        messages.append({"role": "user", "content": force_msg})
        try:
            resp = call_claude_api(messages, api_key, system=DIAGNOSTIC_AGENT_PROMPT,
                                   tools=TOOLS, use_files_beta=(test_file_id is not None))
            for b in resp.get("content", []):
                if b.get("type") == "tool_use" and b.get("name") == "submit_prediction":
                    inp = b.get("input", {})
                    prediction = inp.get("prediction", "UNKNOWN")
                    confidence = float(inp.get("confidence", 0.25))
                    reasoning  = inp.get("reasoning", "forced final prediction")
                    trace.append({"phase": "final_prediction",
                                  "parsed": {"prediction": prediction,
                                             "confidence": confidence,
                                             "reasoning": reasoning},
                                  "raw": "forced submit"})
                    break
            if prediction == "UNKNOWN":
                parsed     = parse_json(extract_text(resp.get("content", [])))
                prediction = parsed.get("prediction",
                                        sorted(expected_classes)[0] if expected_classes
                                        else "UNKNOWN")
                confidence = float(parsed.get("confidence", 0.25))
                reasoning  = parsed.get("reasoning", "forced fallback")
        except Exception as e:
            prediction = sorted(expected_classes)[0] if expected_classes else "UNKNOWN"
            confidence = 0.25
            reasoning  = f"forced fallback after error: {e}"

    if test_file_id:
        delete_file(test_file_id, api_key)

    return {
        "prediction":        prediction,
        "confidence":        confidence,
        "reasoning":         reasoning,
        "trace":             trace,
        "tool_call_count":   tool_call_count,
        "refs_viewed":       refs_viewed,
        "symptoms_read":     symptoms_read,
        "comparisons":       comparisons,
        "success":           prediction != "UNKNOWN",
        "error":             None,
        "num_turns":         len(trace),
        "reasoning_summary": reasoning[:200] if reasoning else "",
        "vision_features":   vision_features,
        "ablation":          ablation,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  JUDGE — Stage 3
# ══════════════════════════════════════════════════════════════════════════════

def run_judge(image_path: str, ground_truth: str, prediction: str,
              confidence: float, reasoning_trace: list, api_key: str) -> dict:
    trace_summary = [
        f"  [{e.get('tool', e.get('phase',''))}]: {e.get('result_text','')[:120]}"
        for e in reasoning_trace[-8:]
    ]
    judge_prompt = (
        f"Ground truth : {ground_truth}\nPrediction   : {prediction}\n"
        f"Confidence   : {confidence:.2f}\nCorrect      : {prediction == ground_truth}\n\n"
        "Agent reasoning trace (last steps):\n"
        + "\n".join(trace_summary) + "\n\n"
        "Evaluate calibration. Respond with JSON only:\n"
        '{\n  "calibration_verdict": "<WELL_CALIBRATED|OVERCONFIDENT|UNDERCONFIDENT|INCONSISTENT>",\n'
        '  "calibration_score": <0.0 to 1.0>,\n'
        '  "reasoning_consistency": "<brief note>",\n'
        '  "judge_notes": "<1-2 sentences on agent performance>"\n}'
    )
    try:
        resp   = call_claude_api(messages=[{"role": "user", "content": judge_prompt}],
                                 api_key=api_key, system=JUDGE_SYSTEM_PROMPT, max_retries=4)
        text   = extract_text(resp.get("content", []))
        parsed = parse_json(text)
        return {"parsed": parsed, "raw": text}
    except Exception as e:
        return {"parsed": {"calibration_verdict": "UNKNOWN", "calibration_score": 0.0,
                           "reasoning_consistency": "", "judge_notes": str(e)}, "raw": ""}


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def _safe_input(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print(); return ""


def _parse_number_list(raw: str, lo: int, hi: int):
    try:
        nums = [int(x.strip()) for x in raw.split(",") if x.strip()]
        if not nums or any(n < lo or n > hi for n in nums):
            return None
        return sorted(set(nums))
    except ValueError:
        return None


def prompt_user_for_datasets(crop_map: dict) -> list:
    crops   = sorted(crop_map.keys())
    n_crops = len(crops)
    if not crops:
        print(f"ERROR: No crops found in {BENCH_DIR}"); return []

    # Two top-level options only
    print("=" * 60)
    print("  RUN MODE")
    print("=" * 60)
    print("  [1]  Both ablations — run images TWICE: once with Local_Registry")
    print("       (doc-extracted KB) and once with Registry (web-enriched KB).")
    print("       Use this to compare accuracy between the two KB sources.")
    print()
    print("  [2]  Single ablation — run images ONCE with the KB source you")
    print(f"       specified via --ablation (currently: {DEFAULT_ABLATION!r}).")
    print()
    while True:
        mode = _safe_input("  Your choice (1 or 2): ")
        if mode in ("1", "2"): break
        print("  Please enter 1 or 2.")
    both_ablations = (mode == "1")

    # Crop selection
    print()
    print("=" * 60)
    print(f"  AVAILABLE CROPS  ({BENCH_DIR.name}/)")
    print("=" * 60)
    for i, crop in enumerate(crops, 1):
        total = sum(count_bench_images(crop, c) for c in crop_map[crop])
        print(f"  [{i:>2}]  {crop:<40}  {len(crop_map[crop])} classes  {total} images")
    print()

    while True:
        raw = _safe_input(f"  Select crop(s) — comma-separated numbers (1-{n_crops}), or 'all': ")
        if not raw: return []
        if raw.strip().lower() == "all":
            selected_crops = crops; break
        nums = _parse_number_list(raw, 1, n_crops)
        if nums:
            selected_crops = [crops[n - 1] for n in nums]
            print(f"  -> Selected crops: {selected_crops}"); break
        print(f"  Enter comma-separated numbers 1-{n_crops}, or 'all'.")

    # Per-crop: disease + image selection
    run_config = []
    for crop in selected_crops:
        classes = crop_map[crop]
        n_cls   = len(classes)
        print()
        print("-" * 60)
        print(f"  CROP: {crop}  ({n_cls} disease classes)")
        print("-" * 60)
        for i, cls in enumerate(classes, 1):
            print(f"    [{i:>3}]  {cls:<50}  ({count_bench_images(crop, cls)} images)")
        print()

        while True:
            raw = _safe_input(f"  Select disease(s) — comma-separated numbers (1-{n_cls}), or 'all': ")
            if not raw: return []
            if raw.strip().lower() == "all":
                selected_classes = None
                print(f"    -> All {n_cls} classes selected."); break
            nums = _parse_number_list(raw, 1, n_cls)
            if nums:
                selected_classes = [classes[n - 1] for n in nums]
                print(f"    -> Selected: {selected_classes}"); break
            print(f"  Enter comma-separated numbers 1-{n_cls}, or 'all'.")

        while True:
            raw = _safe_input("  Images per class? (Enter = all, or a number): ")
            if raw == "":
                ipc = None; print("    -> All images per class."); break
            try:
                ipc = int(raw)
                if ipc > 0:
                    print(f"    -> {ipc} image(s) per class."); break
            except ValueError: pass
            print("  Enter a positive integer or press Enter.")

        if both_ablations:
            for abl in ("local", "web"):
                run_config.append({"crop": crop, "classes": selected_classes,
                                   "images_per_class": ipc, "ablation": abl})
        else:
            run_config.append({"crop": crop, "classes": selected_classes,
                               "images_per_class": ipc})

    return run_config


# ══════════════════════════════════════════════════════════════════════════════
#  CROP RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_agent_on_crop(crop_name: str, logs_dir: Path,
                      symptom_lookup: dict, ablation: str, api_key: str,
                      selected_classes=None, images_per_class=None) -> dict:

    crop_map    = get_crop_class_map()
    all_classes = crop_map.get(crop_name, [])
    if selected_classes:
        all_classes = [c for c in all_classes if c in selected_classes]
    if not all_classes:
        print(f"  No classes found for '{crop_name}'"); return {}

    test_images = []
    for cls in all_classes:
        test_images.extend(get_bench_images(crop_name, cls, images_per_class))
    if not test_images:
        print(f"  No benchmark images for '{crop_name}'"); return {}

    used = []
    if ablation == "local": used.append("local=Local_Registry")
    if ablation == "web":   used.append("web=Registry")

    dataset_description = (
        f"Crop '{crop_name}' — {len(all_classes)} classes, "
        f"{len(test_images)} images  |  KB ablation: {ablation} ({', '.join(used)})"
    )

    print(f"\n{'='*60}")
    print(f"Crop     : {crop_name}")
    print(f"Classes  : {len(all_classes)}")
    print(f"Images   : {len(test_images)}")
    print(f"Ablation : {ablation}  ({', '.join(used)})")
    print(f"{'='*60}")

    crop_logs_dir = logs_dir / crop_name / ablation
    crop_logs_dir.mkdir(parents=True, exist_ok=True)

    correct, total_refs, total_tools = 0, 0, 0
    judge_scores: list = []
    calibration_counts = {
        "WELL_CALIBRATED": 0, "OVERCONFIDENT": 0,
        "UNDERCONFIDENT": 0,  "INCONSISTENT":  0, "UNKNOWN": 0,
    }
    ref_file_cache: dict = {}

    try:
        for idx, (image_path, ground_truth, image_name) in enumerate(test_images, 1):
            print(f"\n[{idx}/{len(test_images)}] {image_name} | GT: {ground_truth}")
            print("  ", end="", flush=True)

            result = classify_with_agent(
                image_path          = image_path,
                expected_classes    = all_classes,
                dataset_description = dataset_description,
                symptom_lookup      = symptom_lookup,
                crop_name           = crop_name,
                ablation            = ablation,
                api_key             = api_key,
                ref_file_cache      = ref_file_cache,
            )

            prediction  = result.get("prediction", "UNKNOWN")
            confidence  = result.get("confidence", 0.0)
            tool_calls  = result.get("tool_call_count", 0)
            refs_viewed = result.get("refs_viewed", [])
            syms_read   = result.get("symptoms_read", [])
            comparisons = result.get("comparisons", [])
            is_correct  = (prediction == ground_truth)

            if is_correct: correct += 1
            total_refs  += len(refs_viewed)
            total_tools += tool_calls

            print(f"\n  GT={ground_truth} | Pred={prediction} | "
                  f"{'✓' if is_correct else '✗'} "
                  f"(conf:{confidence:.2f} tools:{tool_calls})")
            if refs_viewed:  print(f"    Refs viewed  -> {refs_viewed}")
            if syms_read:    print(f"    Syms read    -> {syms_read}")
            if comparisons:  print(f"    Comparisons  -> {comparisons}")

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
            if reason: print(f"    Reason       -> {reason[:200]}")

            print("    Judge        -> ", end="", flush=True)
            judge_result = run_judge(image_path, ground_truth, prediction,
                                     confidence, result["trace"], api_key)
            judge_parsed = judge_result.get("parsed", {})
            verdict      = judge_parsed.get("calibration_verdict", "UNKNOWN")
            cal_score    = float(judge_parsed.get("calibration_score", 0.0))
            judge_notes  = judge_parsed.get("judge_notes", "")

            if verdict in calibration_counts:
                calibration_counts[verdict] += 1
            judge_scores.append(cal_score)
            print(f"{verdict}  (cal:{cal_score:.2f})")
            if judge_notes: print(f"               {judge_notes[:150]}")

            log_entry = {
                "image_name":        image_name,
                "crop":              crop_name,
                "ground_truth":      ground_truth,
                "prediction":        prediction,
                "correct":           is_correct,
                "confidence":        confidence,
                "ablation":          ablation,
                "tool_call_count":   tool_calls,
                "refs_viewed":       refs_viewed,
                "symptoms_read":     syms_read,
                "comparisons_made":  comparisons,
                "vision_features":   vf,
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
            log_file = crop_logs_dir / f"{Path(image_name).stem}_log.json"
            with open(log_file, "w") as f:
                json.dump(log_entry, f, indent=2)
            print()

    finally:
        if ref_file_cache:
            print(f"  Cleaning up {len(ref_file_cache)} cached ref files...", end=" ")
            for fid in ref_file_cache.values():
                if fid: delete_file(fid, api_key)
            print("done")

    n         = len(test_images)
    accuracy  = (correct / n) * 100 if n else 0
    avg_cal   = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0
    avg_refs  = total_refs  / n if n else 0
    avg_tools = total_tools / n if n else 0

    print(f"\n  Accuracy         : {correct}/{n} ({accuracy:.1f}%)")
    print(f"  Avg tool calls   : {avg_tools:.1f}  (ceiling: {MAX_TOOL_CALLS})")
    print(f"  Avg ref images   : {avg_refs:.1f}")
    print(f"  Avg cal. score   : {avg_cal:.2f}")
    print("  Calibration breakdown:")
    for v, c in calibration_counts.items():
        print(f"    {v:<20}: {c}")
    print(f"  Logs             : {crop_logs_dir}/")

    return {
        "crop":                  crop_name,
        "ablation":              ablation,
        "accuracy":              accuracy,
        "avg_calibration_score": avg_cal,
        "avg_ref_turns":         avg_refs,
        "avg_tool_calls":        avg_tools,
        "calibration_counts":    calibration_counts,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_agent():
    print("=" * 60)
    print("AGENTIC CLASSIFICATION  (tool-use + Files API)")
    print(f"  Model         : {MODEL}")
    print(f"  Max tool calls: {MAX_TOOL_CALLS} per image")
    print(f"  Image size    : max {MAX_LONG_EDGE}px, JPEG q{JPEG_QUALITY}")
    print(f"  Bench dir     : {BENCH_DIR}")
    print(f"  Ref image dir : {REF_IMAGE_DIR}")
    print(f"  Registry xlsx : {REGISTRY_PATH.name}")
    print("=" * 60)

    if not CURATED_DIR.exists():
        print(f"\nERROR: {CURATED_DIR} not found."); return
    if not REGISTRY_PATH.exists():
        print(f"\nERROR: {REGISTRY_PATH.name} not found. "
              "Run generate_symptoms.py first."); return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nERROR: ANTHROPIC_API_KEY not set."); return

    print(f"\nLoading KBs from {REGISTRY_PATH.name} …")
    kb_cache: dict = {}
    for abl_mode in VALID_ABLATION_MODES:
        try:
            kb_cache[abl_mode] = load_kb_from_xlsx(REGISTRY_PATH, abl_mode)
            print(f"  {abl_mode:<6}: {len(kb_cache[abl_mode])} entries loaded.")
        except Exception as e:
            print(f"  {abl_mode:<6}: WARNING — {e}")
            kb_cache[abl_mode] = {}

    crop_map = get_crop_class_map()
    if not crop_map:
        print(f"\nERROR: No crops found in {BENCH_DIR}"); return

    run_config = prompt_user_for_datasets(crop_map)
    if not run_config:
        print("No crops selected. Exiting."); return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logs_dir = RESULTS_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict = {}
    for cfg in run_config:
        crop         = cfg["crop"]
        cfg_ablation = cfg.get("ablation", DEFAULT_ABLATION)
        symptom_lookup = kb_cache.get(cfg_ablation, {})
        stats = run_agent_on_crop(
            crop_name        = crop,
            logs_dir         = logs_dir,
            symptom_lookup   = symptom_lookup,
            ablation         = cfg_ablation,
            api_key          = api_key,
            selected_classes = cfg.get("classes"),
            images_per_class = cfg.get("images_per_class"),
        )
        all_results[f"{crop}_{cfg_ablation}"] = stats

    print("\n" + "=" * 60)
    print("GLOBAL SUMMARY")
    print("=" * 60)
    for crop, stats in all_results.items():
        if not stats: continue
        counts = stats["calibration_counts"]
        print(f"  {crop}")
        print(f"    Accuracy       : {stats['accuracy']:.1f}%")
        print(f"    Avg tool calls : {stats['avg_tool_calls']:.1f}")
        print(f"    Avg ref images : {stats['avg_ref_turns']:.1f}")
        print(f"    Avg cal. score : {stats['avg_calibration_score']:.2f}")
        print(f"    Calibration    : "
              f"WELL={counts['WELL_CALIBRATED']} "
              f"OVER={counts['OVERCONFIDENT']} "
              f"UNDER={counts['UNDERCONFIDENT']} "
              f"INCON={counts['INCONSISTENT']}")

    print(f"\nFull logs: {logs_dir}/")
    return all_results



# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_agent()