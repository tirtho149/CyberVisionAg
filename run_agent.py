#!/usr/bin/env python3
"""
Truly agentic plant disease classification — with Files API.

Images are uploaded ONCE via the Anthropic Files API and referenced by
file_id in every subsequent API call. This eliminates the multi-MB base64
payloads that caused 502/connection-drop errors.

Upload strategy
───────────────
  • Test image    : uploaded once at the start of each image's classification,
                    deleted after judge scoring is complete.
  • Reference imgs: uploaded once per class per dataset run, cached in a dict
                    {class_name → file_id}, deleted at the end of the dataset run.

All Messages API calls use:
    {"type": "image", "source": {"type": "file", "file_id": "file_xxx"}}
instead of base64 inline blocks — payload is now a few bytes per image.

Tools available to the agent
─────────────────────────────
  read_symptom_description(disease_name)  → symptom text from knowledge base
  get_reference_image(class_name)         → file_id reference image from train split
  list_dataset_classes()                  → full class list
  submit_prediction(prediction, confidence, reasoning) → ends the loop
"""

import os
import io
import json
import base64
import re
import time
from pathlib import Path
from datetime import datetime
from dotenv import dotenv_values
import requests

# ── Environment ────────────────────────────────────────────────────────────────
env_file = Path(__file__).parent / ".env"
env_vars = dotenv_values(env_file)
for key, value in env_vars.items():
    if value:
        os.environ[key] = value

# ── Paths & constants ──────────────────────────────────────────────────────────
CURATED_DIR   = Path(__file__).parent / "Curated_Local_Dataset"
TRAIN_DIR     = CURATED_DIR / "train"
TEST_DIR      = CURATED_DIR / "test"
RESULTS_DIR   = Path(__file__).parent / "results" / "agent"
SYMPTOMS_FILE = Path(__file__).parent / "disease_symptoms_crop_wise.md"

MODEL          = "claude-sonnet-4-6"
MAX_TOKENS     = 4096
MAX_TOOL_CALLS = 12
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# Target longest edge in pixels — enough for disease pattern recognition
MAX_LONG_EDGE  = 1024
JPEG_QUALITY   = 85

FILES_API_BETA = "files-api-2025-04-14"
FILES_BASE_URL = "https://api.anthropic.com/v1/files"
MESSAGES_URL   = "https://api.anthropic.com/v1/messages"

RETRYABLE_STATUS_CODES = {429, 502, 503, 504}

# ── System prompts ─────────────────────────────────────────────────────────────
AGENT_SYSTEM_PROMPT = """You are an expert plant pathologist classifying plant diseases from images.

You have access to the following tools:
  read_symptom_description  — look up visual symptom details for any disease class
  get_reference_image       — fetch a confirmed training example to compare visually
  list_dataset_classes      — remind yourself of all available classes
  submit_prediction         — commit your final answer (MUST be called to finish)

Your workflow:
1. Examine the target image carefully
2. Use read_symptom_description for your top candidate classes
3. Use get_reference_image to visually confirm your top candidate(s)
4. Call submit_prediction when you are confident

Rules:
  • You MUST call submit_prediction — never refuse to classify
  • prediction must be exactly one of the listed classes
  • confidence is 0.0 to 1.0; be honest
  • You decide how many tools to call"""

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing AI plant disease classification agents.
Evaluate ONLY confidence calibration: did the agent's stated confidence match the actual outcome?

Calibration verdicts:
  WELL_CALIBRATED  — confidence matched outcome (high+correct or low+wrong)
  OVERCONFIDENT    — stated high confidence (>0.7) but was incorrect
  UNDERCONFIDENT   — stated low confidence (<0.4) but was correct
  INCONSISTENT     — confidence did not reflect the quality of reasoning shown

Respond ONLY with valid JSON matching the schema in the prompt."""

# ── Tool schemas ───────────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "read_symptom_description",
        "description": (
            "Look up the visual symptom description for a specific disease class "
            "from the knowledge base. Use this to compare what you see in the image "
            "against the expected symptoms for a candidate class."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "disease_name": {
                    "type": "string",
                    "description": "Exact class name, e.g. 'Gray_leaf_spot'"
                }
            },
            "required": ["disease_name"]
        }
    },
    {
        "name": "get_reference_image",
        "description": (
            "Fetch a confirmed training example image for a given class. "
            "The image will be returned so you can compare it visually with the target."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "class_name": {
                    "type": "string",
                    "description": "Exact class name, e.g. 'Anthracnose'"
                }
            },
            "required": ["class_name"]
        }
    },
    {
        "name": "list_dataset_classes",
        "description": "Return the full list of disease classes available in this dataset.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "submit_prediction",
        "description": (
            "Submit your final classification. Call this when ready to commit. "
            "This ends the agent loop."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prediction": {
                    "type": "string",
                    "description": "Predicted class — must be exactly one of the available classes"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence 0.0 (uncertain) to 1.0 (certain)"
                },
                "reasoning": {
                    "type": "string",
                    "description": "1-3 sentences explaining the visual evidence"
                }
            },
            "required": ["prediction", "confidence", "reasoning"]
        }
    }
]


# ── Knowledge-base helpers ─────────────────────────────────────────────────────

def load_symptoms() -> str:
    with open(SYMPTOMS_FILE, "r") as f:
        return f.read()


def parse_symptoms_by_crop(symptoms_text: str) -> dict:
    parsed: dict = {}
    current_crop = ""
    current_disease = ""
    collecting = False
    buf: list = []

    for line in symptoms_text.splitlines():
        crop_m = re.match(r"^##\s+Crop:\s+(.+)$", line)
        if crop_m:
            if current_crop and current_disease and buf:
                parsed.setdefault(current_crop, {})[current_disease] = " ".join(buf).strip()
            current_crop    = crop_m.group(1).strip()
            current_disease = ""
            buf             = []
            collecting      = False
            parsed.setdefault(current_crop, {})
            continue

        dis_m = re.match(r"^###\s+(.+)$", line)
        if dis_m:
            if current_crop and current_disease and buf:
                parsed.setdefault(current_crop, {})[current_disease] = " ".join(buf).strip()
            current_disease = dis_m.group(1).strip().replace(" ", "_")
            buf             = []
            collecting      = False
            continue

        if line.strip() == "**Symptoms:**":
            collecting = True
            buf        = []
            continue
        if line.strip() == "**Reference Images:**":
            collecting = False
            continue
        if collecting and line.strip() and not line.startswith("-"):
            buf.append(line.strip())

    if current_crop and current_disease and buf:
        parsed.setdefault(current_crop, {})[current_disease] = " ".join(buf).strip()
    return parsed


def build_symptom_lookup(symptoms_text: str, dataset_name: str) -> dict:
    all_syms = parse_symptoms_by_crop(symptoms_text)
    norm     = dataset_name.replace("_", " ").lower()
    matched  = None
    for k in all_syms:
        if k.lower() == norm:
            matched = k
            break
    if not matched:
        for k in all_syms:
            if norm in k.lower() or k.lower() in norm:
                matched = k
                break
    return all_syms.get(matched, {})


# ── Image compression ──────────────────────────────────────────────────────────

def compress_image_to_bytes(image_path: str) -> tuple:
    """
    Always convert to JPEG, cap longest edge at MAX_LONG_EDGE px.
    Returns (jpeg_bytes, "image/jpeg").
    Falls back to raw file bytes if Pillow is not installed.
    """
    try:
        from PIL import Image as PILImage

        img = PILImage.open(image_path).convert("RGB")

        # Resize so longest edge <= MAX_LONG_EDGE
        w, h     = img.size
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
        # Pillow not available — send raw
        raw = Path(image_path).read_bytes()
        ext = Path(image_path).suffix.lower()
        mt  = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".png": "image/png",  ".webp": "image/webp"}.get(ext, "image/jpeg")
        return raw, mt


# ── Files API helpers ──────────────────────────────────────────────────────────

def _files_headers(api_key: str) -> dict:
    return {
        "x-api-key":        api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta":    FILES_API_BETA,
    }


def upload_image(image_path: str, api_key: str,
                 max_retries: int = 8):
    """
    Compress the image, upload via Files API, return file_id string.
    Returns None on total failure instead of raising — callers fall back
    to inline base64 automatically.
    """
    img_bytes, mime_type = compress_image_to_bytes(image_path)
    filename = Path(image_path).stem + ".jpg"

    wait = 10
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                FILES_BASE_URL,
                headers=_files_headers(api_key),
                files={"file": (filename, img_bytes, mime_type)},
                timeout=60,
            )
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                print(f"\n    [upload conn error] {e.__class__.__name__} "
                      f"– retrying in {wait}s...", end="", flush=True)
                time.sleep(wait)
                wait = min(wait * 2, 60)
                continue
            print(f"\n    [upload failed] {e.__class__.__name__} after {max_retries} attempts "
                  f"– falling back to inline base64", flush=True)
            return None

        if resp.status_code == 200:
            file_id = resp.json().get("id", "")
            if file_id:
                return file_id
            print("\n    [upload warn] Files API returned no id – falling back to inline base64",
                  flush=True)
            return None

        if resp.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
            wait_for = int(resp.headers.get("retry-after", wait))
            print(f"\n    [upload {resp.status_code}] waiting {wait_for}s...",
                  end="", flush=True)
            time.sleep(wait_for)
            wait = min(wait * 2, 60)
            continue

        print(f"\n    [upload error] {resp.status_code} – falling back to inline base64"
              f"\n    [upload response] {resp.text[:200]}", flush=True)
        return None

    print("\n    [upload exhausted] – falling back to inline base64", flush=True)
    return None


def delete_file(file_id: str, api_key: str) -> None:
    """Best-effort delete — errors are logged but not raised."""
    try:
        requests.delete(
            f"{FILES_BASE_URL}/{file_id}",
            headers=_files_headers(api_key),
            timeout=15,
        )
    except Exception as e:
        print(f"\n    [warn] Could not delete file {file_id}: {e}")


def file_image_block(file_id: str) -> dict:
    """Content block that references an already-uploaded file by ID."""
    return {
        "type": "image",
        "source": {
            "type":    "file",
            "file_id": file_id,
        }
    }


def inline_image_block(image_path: str) -> dict:
    """Fallback: compress and base64-encode inline when Files API is unavailable."""
    img_bytes, mime_type = compress_image_to_bytes(image_path)
    data = base64.standard_b64encode(img_bytes).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": mime_type, "data": data},
    }


def image_content_block(image_path: str, api_key: str,
                         file_id_or_none) -> tuple:
    """
    Return (content_block, file_id_or_none).
    If file_id_or_none is provided, use Files API reference.
    If it is None (upload failed), use compressed inline base64.
    Also handles the case where we need to upload fresh.
    """
    if file_id_or_none:
        return file_image_block(file_id_or_none), file_id_or_none
    # Try a fresh upload
    fid = upload_image(image_path, api_key)
    if fid:
        return file_image_block(fid), fid
    # Fallback: inline base64
    print("[inline-fallback]", end=" ", flush=True)
    return inline_image_block(image_path), None


# ── Utility headers for Messages API (needs beta header too) ───────────────────

def _messages_headers(api_key: str, include_files_beta: bool = False) -> dict:
    h = {
        "x-api-key":        api_key,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }
    if include_files_beta:
        h["anthropic-beta"] = FILES_API_BETA
    return h


# ── Messages API call (with retry) ────────────────────────────────────────────

def call_claude_api(messages: list, api_key: str,
                    system: str, tools: list = None,
                    max_retries: int = 8,
                    use_files_beta: bool = False) -> dict:
    """
    POST /v1/messages.
    use_files_beta=True adds the Files API beta header (needed when messages
    contain file_id image references). Falls back cleanly without it.
    Retries on 429/502/503/504 and connection-level errors.
    """
    payload: dict = {
        "model":      MODEL,
        "max_tokens": MAX_TOKENS,
        "system":     system,
        "messages":   messages,
    }
    if tools:
        payload["tools"] = tools

    wait = 15
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                MESSAGES_URL,
                headers=_messages_headers(api_key, include_files_beta=use_files_beta),
                json=payload,
                timeout=180,
            )
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout) as conn_err:
            if attempt < max_retries - 1:
                print(f"\n    [connection error] {conn_err.__class__.__name__} "
                      f"– waiting {wait}s (attempt {attempt+1}/{max_retries})...",
                      end="", flush=True)
                time.sleep(wait)
                wait = min(wait * 2, 120)
                continue
            raise RuntimeError(f"Connection error after {max_retries} attempts: {conn_err}")

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
            retry_after = resp.headers.get("retry-after")
            sleep       = int(retry_after) if retry_after else wait
            label       = "rate-limit" if resp.status_code == 429 else "server error"
            print(f"\n    [{label}] {resp.status_code} – waiting {sleep}s "
                  f"(attempt {attempt+1}/{max_retries})...", end="", flush=True)
            time.sleep(sleep)
            wait = min(wait * 2, 120)
            continue

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


# ── Reference image path finder ────────────────────────────────────────────────

def find_reference_image(class_name: str, dataset_name: str):
    class_dir = TRAIN_DIR / dataset_name / class_name
    if not class_dir.exists():
        return None
    for pat in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        found = sorted(class_dir.glob(pat))
        if found:
            return str(found[0])
    return None


# ── Tool executor ──────────────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: dict,
                 dataset_name: str, expected_classes: list,
                 symptom_lookup: dict,
                 ref_file_cache: dict,
                 api_key: str) -> tuple:
    """
    Execute a tool call and return (result_content_blocks, trace_meta_dict).

    ref_file_cache: {class_name: file_id} — shared across all tool calls
    in a single image's agent loop so each reference class is uploaded at most once.
    """
    if tool_name == "list_dataset_classes":
        text = (
            f"Available classes in {dataset_name} ({len(expected_classes)} total):\n"
            + "\n".join(f"  • {c}" for c in expected_classes)
        )
        return [{"type": "text", "text": text}], {"tool": tool_name}

    if tool_name == "read_symptom_description":
        disease = tool_input.get("disease_name", "")
        key     = disease.replace(" ", "_")
        symptom = symptom_lookup.get(key)
        if not symptom:
            lower   = {k.lower(): v for k, v in symptom_lookup.items()}
            symptom = lower.get(key.lower())
        text = (
            f"[{disease}] Visual symptom description:\n{symptom}"
            if symptom else
            f"No entry found for '{disease}'. Available: {list(symptom_lookup.keys())[:10]}"
        )
        return [{"type": "text", "text": text}], {"tool": tool_name, "disease": disease}

    if tool_name == "get_reference_image":
        cls      = tool_input.get("class_name", "")
        ref_path = find_reference_image(cls, dataset_name)

        if not ref_path:
            text = f"No reference image found for '{cls}'. Use read_symptom_description instead."
            return [{"type": "text", "text": text}], {
                "tool": tool_name, "class": cls, "ref_image": None, "file_id": None
            }

        # Upload once, reuse file_id for subsequent calls to the same class
        # None is stored on failure so we don't retry every time
        if cls not in ref_file_cache:
            fid = upload_image(ref_path, api_key)   # returns None on failure
            ref_file_cache[cls] = fid               # cache None too

        file_id = ref_file_cache[cls]
        img_block = file_image_block(file_id) if file_id else inline_image_block(ref_path)
        blocks  = [
            img_block,
            {"type": "text", "text": f"Reference training image for class '{cls}'."},
        ]
        return blocks, {"tool": tool_name, "class": cls,
                        "ref_image": ref_path, "file_id": file_id}

    if tool_name == "submit_prediction":
        prediction = tool_input.get("prediction", "UNKNOWN")
        confidence = float(tool_input.get("confidence", 0.0))
        reasoning  = tool_input.get("reasoning", "")
        text       = f"Prediction submitted: {prediction} (confidence={confidence:.2f})"
        return [{"type": "text", "text": text}], {
            "tool": tool_name, "prediction": prediction,
            "confidence": confidence, "reasoning": reasoning,
        }

    return [{"type": "text", "text": f"Unknown tool: {tool_name}"}], {"tool": tool_name}


# ── Core agentic loop ──────────────────────────────────────────────────────────

def classify_with_agent(
    image_path: str,
    expected_classes: list,
    dataset_description: str,
    symptoms_text: str,
    dataset_name: str,
    api_key: str,
    ref_file_cache: dict,        # shared across all images in this dataset run
) -> dict:
    """
    Upload the test image once, then let Claude drive classification via tools.
    ref_file_cache is passed in so reference images are uploaded at most once
    per class across the whole dataset run.
    """
    symptom_lookup = build_symptom_lookup(symptoms_text, dataset_name)

    # ── Upload test image (fallback to inline base64 if upload fails) ───────────
    test_file_id = upload_image(image_path, api_key)
    if test_file_id:
        print(f"[file-api]", end=" ", flush=True)
        test_img_block = file_image_block(test_file_id)
    else:
        print(f"[inline-b64]", end=" ", flush=True)
        test_img_block = inline_image_block(image_path)

    initial_prompt = (
        "Please classify the plant disease shown in the image above.\n\n"
        f"Dataset: {dataset_description}\n"
        f"Available classes: {expected_classes}\n\n"
        "Use your tools to investigate:\n"
        "  1. read_symptom_description for your top candidate classes\n"
        "  2. get_reference_image to visually confirm\n"
        "  3. submit_prediction when confident\n\n"
        "You MUST call submit_prediction with a class from the list above."
    )

    messages = [{
        "role": "user",
        "content": [
            test_img_block,
            {"type": "text", "text": initial_prompt},
        ]
    }]

    trace           = []
    tool_call_count = 0
    prediction      = "UNKNOWN"
    confidence      = 0.0
    reasoning       = ""
    refs_viewed     = []
    symptoms_read   = []
    submitted       = False

    # ── Agentic tool-use loop ──────────────────────────────────────────────────
    while tool_call_count < MAX_TOOL_CALLS and not submitted:
        try:
            resp = call_claude_api(messages, api_key,
                                   system=AGENT_SYSTEM_PROMPT,
                                   tools=TOOLS,
                                   use_files_beta=(test_file_id is not None))
        except Exception as e:
            print(f"\n    [API ERROR] {e}", flush=True)
            if test_file_id:
                delete_file(test_file_id, api_key)
            return {
                "prediction": "ERROR", "confidence": 0.0,
                "trace": trace, "error": str(e), "success": False,
                "num_turns": tool_call_count, "refs_viewed": refs_viewed,
                "ref_turns_used": len(refs_viewed), "reasoning_summary": "",
                "symptoms_read": symptoms_read, "tool_call_count": tool_call_count,
            }

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
                trace.append({"phase": "freetext_prediction",
                               "raw": assistant_text, "parsed": parsed})
            else:
                trace.append({"phase": "no_tool_no_json", "raw": assistant_text})
            print(f"[stop:{stop_reason}]", end=" ", flush=True)
            break

        tool_result_blocks = []

        for tu in tool_uses:
            tool_name  = tu.get("name", "")
            tool_input = tu.get("input", {})
            tool_id    = tu.get("id", "")
            tool_call_count += 1

            result_blocks, meta = execute_tool(
                tool_name, tool_input,
                dataset_name, expected_classes, symptom_lookup,
                ref_file_cache, api_key,
            )

            trace.append({
                "phase":       "tool_call",
                "tool":        tool_name,
                "input":       tool_input,
                "tool_id":     tool_id,
                "meta":        meta,
                "result_text": extract_text(result_blocks),
            })

            if tool_name == "get_reference_image":
                cls = tool_input.get("class_name", "")
                if cls and cls not in refs_viewed:
                    refs_viewed.append(cls)
                print(f"[ref:{cls[:8]}]", end=" ", flush=True)
            elif tool_name == "read_symptom_description":
                dis = tool_input.get("disease_name", "")
                if dis and dis not in symptoms_read:
                    symptoms_read.append(dis)
                print(f"[sym:{dis[:8]}]", end=" ", flush=True)
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
                               "confidence": confidence,
                               "reasoning":  reasoning},
                    "raw": "submit_prediction called",
                })
                print("[submit]", end=" ", flush=True)

            tool_result_blocks.append({
                "type":        "tool_result",
                "tool_use_id": tool_id,
                "content":     result_blocks,
            })

        messages.append({"role": "user", "content": tool_result_blocks})

        if submitted:
            break

    # ── Safety net ────────────────────────────────────────────────────────────
    if prediction == "UNKNOWN":
        print("[force-final]", end=" ", flush=True)
        force_msg = (
            f"You have used {tool_call_count} tool calls but have not called "
            f"submit_prediction. You MUST call submit_prediction NOW with your best "
            f"guess from: {expected_classes}"
        )
        messages.append({"role": "user",
                          "content": [{"type": "text", "text": force_msg}]})
        try:
            resp2   = call_claude_api(messages, api_key,
                                      system=AGENT_SYSTEM_PROMPT, tools=TOOLS,
                                      use_files_beta=(test_file_id is not None))
            blocks2 = resp2.get("content", [])
            messages.append({"role": "assistant", "content": blocks2})
            for tu in (b for b in blocks2 if b.get("type") == "tool_use"):
                if tu.get("name") == "submit_prediction":
                    inp        = tu.get("input", {})
                    prediction = inp.get("prediction", "UNKNOWN")
                    confidence = float(inp.get("confidence", 0.0))
                    reasoning  = inp.get("reasoning", "")
                    trace.append({
                        "phase": "final_prediction",
                        "parsed": {"prediction": prediction,
                                   "confidence": confidence,
                                   "reasoning":  reasoning},
                        "raw": "forced submit_prediction",
                    })
                    break
            if prediction == "UNKNOWN":
                txt    = extract_text(blocks2)
                parsed = parse_json(txt)
                prediction = parsed.get("prediction", "UNKNOWN")
                confidence = float(parsed.get("confidence", 0.0))
                reasoning  = parsed.get("reasoning", txt[:200])
        except Exception as e:
            trace.append({"phase": "force_final_error", "error": str(e)})

    # ── Cleanup test image ─────────────────────────────────────────────────────
    if test_file_id:
        delete_file(test_file_id, api_key)

    # ── Normalise prediction ───────────────────────────────────────────────────
    if prediction not in expected_classes:
        lower_map  = {c.lower(): c for c in expected_classes}
        prediction = lower_map.get(str(prediction).lower(), "UNKNOWN")

    final_entry = next(
        (e for e in reversed(trace) if e.get("phase") == "final_prediction"), {}
    )
    reasoning_summary = (
        f"Symptoms read: {symptoms_read} | "
        f"Refs viewed: {refs_viewed} | "
        f"Total tool calls: {tool_call_count}\n"
        f"Final reasoning: {final_entry.get('parsed', {}).get('reasoning', reasoning)}"
    )

    return {
        "prediction":        prediction,
        "confidence":        confidence,
        "reasoning_summary": reasoning_summary,
        "ref_turns_used":    len(refs_viewed),
        "refs_viewed":       refs_viewed,
        "symptoms_read":     symptoms_read,
        "tool_call_count":   tool_call_count,
        "trace":             trace,
        "num_turns":         tool_call_count,
        "success":           prediction != "UNKNOWN",
    }


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
        phase = entry.get("phase", "")
        if phase == "tool_call":
            trace_summary += (
                f"\n[tool_call] {entry.get('tool')} "
                f"input={entry.get('input')} "
                f"result={entry.get('result_text', '')[:120]}\n"
            )
        elif phase == "final_prediction":
            p = entry.get("parsed", {})
            trace_summary += (
                f"\n[final] prediction={p.get('prediction')} "
                f"confidence={p.get('confidence')} "
                f"reasoning={p.get('reasoning', '')}\n"
            )

    judge_prompt = (
        "CONFIDENCE CALIBRATION EVALUATION\n\n"
        "Target image shown above.\n\n"
        "--- OUTCOME ---\n"
        f"Ground truth      : {ground_truth}\n"
        f"Prediction        : {prediction}\n"
        f"Result            : {'CORRECT' if is_correct else 'INCORRECT'}\n"
        f"Stated confidence : {confidence:.2f}\n\n"
        "--- AGENT TOOL TRACE ---\n"
        f"{trace_summary}\n\n"
        "--- YOUR TASK ---\n"
        f"Evaluate calibration of stated confidence {confidence:.2f}.\n"
        "calibration_score rubric: 1.0=perfect  0.7=mostly  0.4=partial  0.1=bad\n\n"
        "Respond ONLY with this JSON:\n"
        "{\n"
        '  "outcome": "' + ("CORRECT" if is_correct else "INCORRECT") + '",\n'
        f'  "stated_confidence": {confidence:.2f},\n'
        '  "calibration_verdict": "<WELL_CALIBRATED|OVERCONFIDENT|UNDERCONFIDENT|INCONSISTENT>",\n'
        '  "calibration_score": 0.0,\n'
        '  "reasoning_consistency": "<1 sentence>",\n'
        '  "judge_notes": "<1 sentence>"\n'
        "}"
    )

    # Upload the test image for the judge (fallback to inline if upload fails)
    try:
        judge_file_id = upload_image(image_path, api_key)
        if judge_file_id:
            img_block = file_image_block(judge_file_id)
        else:
            img_block = inline_image_block(image_path)

        judge_content = [
            img_block,
            {"type": "text", "text": judge_prompt},
        ]
        resp   = call_claude_api(
            [{"role": "user", "content": judge_content}],
            api_key, system=JUDGE_SYSTEM_PROMPT,
            use_files_beta=(judge_file_id is not None),
        )
        text   = extract_text(resp.get("content", []))
        parsed = parse_json(text)
        if judge_file_id:
            delete_file(judge_file_id, api_key)
        return {"raw": text, "parsed": parsed, "success": True}
    except Exception as e:
        return {"raw": "", "parsed": {}, "error": str(e), "success": False}


# ── Dataset helpers ────────────────────────────────────────────────────────────

def load_curated_test_images(dataset_name: str,
                              selected_classes: list = None,
                              images_per_class: int = None) -> tuple:
    import random
    cat_dir = TEST_DIR / dataset_name
    if not cat_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {cat_dir}")
    all_classes    = sorted([d.name for d in cat_dir.iterdir() if d.is_dir()])
    active_classes = selected_classes if selected_classes else all_classes
    images = []
    for cls in active_classes:
        cls_dir   = cat_dir / cls
        if not cls_dir.exists():
            continue
        img_files = sorted(f for f in cls_dir.iterdir()
                           if f.suffix.lower() in IMAGE_EXTS)
        random.seed(42)
        random.shuffle(img_files)
        if images_per_class:
            img_files = img_files[:images_per_class]
        for img_file in img_files:
            images.append({"name": img_file.name, "path": str(img_file),
                           "ground_truth": cls})
    description = (
        f"{dataset_name.replace('_', ' ')} plant disease classification dataset. "
        f"All possible classes: {all_classes}"
    )
    return images, all_classes, description


# ── Interactive dataset selection ──────────────────────────────────────────────

def get_curated_datasets() -> list:
    if not TEST_DIR.exists():
        return []
    return sorted([d.name for d in TEST_DIR.iterdir() if d.is_dir()])


def get_classes_for_dataset(dataset_name: str) -> list:
    cat_dir = TEST_DIR / dataset_name
    if not cat_dir.exists():
        return []
    return sorted([d.name for d in cat_dir.iterdir() if d.is_dir()])


def count_images_in_class(dataset_name: str, class_name: str) -> int:
    cls_dir = TEST_DIR / dataset_name / class_name
    return len([f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS])


def _ask_int(prompt: str, lo: int, hi: int, zero_means_all: bool = False):
    while True:
        raw = input(prompt).strip()
        try:
            val = int(raw)
            if zero_means_all and val == 0:
                return None
            if lo <= val <= hi:
                return val
            bound = f"0-{hi}" if zero_means_all else f"{lo}-{hi}"
            print(f"  Please enter a number in [{bound}].")
        except ValueError:
            print("  Please enter a valid integer.")


def _ask_indices(prompt: str, available: list, exact_count: int) -> list:
    while True:
        raw   = input(prompt).strip()
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
    available = get_curated_datasets()
    if not available:
        print(f"ERROR: No datasets found in {TEST_DIR}")
        return []

    print("\n" + "=" * 60)
    print("AVAILABLE DATASETS  (Curated_Local_Dataset/test/)")
    print("=" * 60)
    for i, ds in enumerate(available, 1):
        classes  = get_classes_for_dataset(ds)
        n_images = sum(count_images_in_class(ds, c) for c in classes)
        print(f"  [{i}] {ds}  ({len(classes)} classes, {n_images} images total)")

    print()
    n_ds = _ask_int(f"How many datasets? (1-{len(available)}): ", lo=1, hi=len(available))
    if n_ds == len(available):
        selected_datasets = list(available)
        print(f"  -> All {len(available)} datasets selected.")
    else:
        print(f"\nEnter the numbers of the {n_ds} dataset(s) (comma-separated):")
        selected_datasets = _ask_indices("Your choice: ", available, n_ds)
        print(f"  -> Selected: {selected_datasets}")

    run_config = []
    for ds in selected_datasets:
        classes = get_classes_for_dataset(ds)
        print(f"\n{'-'*60}")
        print(f"  DATASET: {ds}  ({len(classes)} classes available)")
        print(f"{'-'*60}")
        for j, cls in enumerate(classes, 1):
            n = count_images_in_class(ds, cls)
            print(f"    [{j}] {cls}  ({n} test images)")

        n_cls = _ask_int(
            f"\n  How many classes from '{ds}'? (1-{len(classes)}, or {len(classes)} for all): ",
            lo=1, hi=len(classes)
        )
        if n_cls == len(classes):
            selected_classes = list(classes)
            print(f"  -> All {len(classes)} classes selected.")
        else:
            print(f"  Enter the numbers of the {n_cls} class(es) (comma-separated):")
            selected_classes = _ask_indices("  Your choice: ", classes, n_cls)
            print(f"  -> Selected classes: {selected_classes}")

        max_imgs = max(count_images_in_class(ds, c) for c in selected_classes)
        imgs_per_class = _ask_int(
            f"  How many test images per class? (1-{max_imgs}, or 0 for all): ",
            lo=1, hi=max_imgs, zero_means_all=True
        )
        print(f"  -> {'All available' if imgs_per_class is None else imgs_per_class} image(s) per class.")

        run_config.append({
            "dataset":          ds,
            "classes":          selected_classes,
            "images_per_class": imgs_per_class,
        })
    return run_config


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
    active = selected_classes if selected_classes else expected_classes
    print(f"Description    : {description}")
    print(f"Testing classes: {active}")
    print(f"Images/class   : {'all' if not images_per_class else images_per_class}")
    print(f"Total images   : {len(test_images)}")
    print(f"Model          : {MODEL}  |  max tool calls: {MAX_TOOL_CALLS}")
    print(f"Reference src  : {TRAIN_DIR / dataset_name}")
    print(f"Image strategy : Files API (upload-once, reference by file_id)\n")

    dataset_logs_dir = logs_dir / dataset_name
    dataset_logs_dir.mkdir(parents=True, exist_ok=True)

    # Shared cache: reference images uploaded once per class per dataset run
    ref_file_cache: dict = {}

    correct     = 0
    total_refs  = 0
    total_tools = 0
    judge_scores = []
    calibration_counts = {
        "WELL_CALIBRATED": 0, "OVERCONFIDENT": 0,
        "UNDERCONFIDENT": 0,  "INCONSISTENT": 0,
    }

    try:
        for i, img_info in enumerate(test_images):
            image_name   = img_info["name"]
            image_path   = img_info["path"]
            ground_truth = img_info["ground_truth"]

            if i > 0:
                time.sleep(3)

            print(f"  [{i+1}/{len(test_images)}] {image_name}  (gt: {ground_truth})")
            print(f"    Agent -> ", end="", flush=True)

            result = classify_with_agent(
                image_path=image_path,
                expected_classes=expected_classes,
                dataset_description=description,
                symptoms_text=symptoms_text,
                dataset_name=dataset_name,
                api_key=api_key,
                ref_file_cache=ref_file_cache,
            )

            prediction  = result["prediction"]
            confidence  = result.get("confidence", 0.0)
            refs_viewed = result.get("refs_viewed", [])
            syms_read   = result.get("symptoms_read", [])
            tool_calls  = result.get("tool_call_count", 0)
            is_correct  = prediction == ground_truth

            if is_correct:
                correct += 1
            total_refs  += len(refs_viewed)
            total_tools += tool_calls

            outcome_str = "CORRECT" if is_correct else f"WRONG -> {prediction}"
            print(f"{'v' if is_correct else 'x'} {outcome_str}  "
                  f"(conf: {confidence:.2f}, tools: {tool_calls})")
            if refs_viewed:
                print(f"    Refs viewed  -> {refs_viewed}")
            if syms_read:
                print(f"    Syms read    -> {syms_read}")

            final_entry = next(
                (e for e in reversed(result["trace"]) if e.get("phase") == "final_prediction"), {}
            )
            reason = final_entry.get("parsed", {}).get("reasoning", "")
            if reason:
                print(f"    Reason -> {reason[:150]}")

            print(f"    Judge  -> ", end="", flush=True)
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
                "tool_call_count":   tool_calls,
                "refs_viewed":       refs_viewed,
                "symptoms_read":     syms_read,
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

    finally:
        # Always clean up cached reference image files from the Files API
        if ref_file_cache:
            print(f"  Cleaning up {len(ref_file_cache)} cached reference file(s)...",
                  end=" ", flush=True)
            for cls, fid in ref_file_cache.items():
                delete_file(fid, api_key)
            print("done")

    n         = len(test_images)
    accuracy  = (correct / n) * 100 if n else 0
    avg_cal   = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0
    avg_refs  = total_refs  / n if n else 0
    avg_tools = total_tools / n if n else 0

    print(f"  Accuracy         : {correct}/{n} ({accuracy:.1f}%)")
    print(f"  Avg tool calls   : {avg_tools:.1f}  (ceiling: {MAX_TOOL_CALLS})")
    print(f"  Avg ref images   : {avg_refs:.1f}")
    print(f"  Avg cal. score   : {avg_cal:.2f}")
    print("  Calibration breakdown:")
    for v, c in calibration_counts.items():
        print(f"    {v:<20}: {c}")
    print(f"  Logs             : {dataset_logs_dir}/")

    return {
        "accuracy":              accuracy,
        "avg_calibration_score": avg_cal,
        "avg_ref_turns":         avg_refs,
        "avg_tool_calls":        avg_tools,
        "calibration_counts":    calibration_counts,
    }


# ── Main entrypoint ────────────────────────────────────────────────────────────

def run_agent(run_config=None):
    print("=" * 60)
    print("AGENTIC CLASSIFICATION  (tool-use + Files API)")
    print(f"  Model         : {MODEL}")
    print(f"  Max tool calls: {MAX_TOOL_CALLS} per image")
    print(f"  Image size    : max {MAX_LONG_EDGE}px long edge, JPEG q{JPEG_QUALITY}")
    print("  Image upload  : Files API — upload once, reference by file_id")
    print("  Tools         : read_symptom_description, get_reference_image,")
    print("                  list_dataset_classes, submit_prediction")
    print("=" * 60)

    if not CURATED_DIR.exists():
        print(f"\nERROR: Curated dataset not found: {CURATED_DIR}")
        print("Run 'python dataloader.py' first.")
        return
    if not SYMPTOMS_FILE.exists():
        print(f"\nERROR: Symptoms file not found: {SYMPTOMS_FILE}")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nERROR: ANTHROPIC_API_KEY not set.")
        return

    if run_config is None:
        run_config = prompt_user_for_datasets()
        if not run_config:
            return

    symptoms_text = load_symptoms()
    print(f"\nKnowledge base   : {SYMPTOMS_FILE} ({len(symptoms_text):,} chars)")
    print(f"Reference images : {TRAIN_DIR}")
    print(f"Max tool calls   : {MAX_TOOL_CALLS}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logs_dir = RESULTS_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for cfg in run_config:
        stats = run_agent_on_dataset(
            dataset_name=cfg["dataset"],
            logs_dir=logs_dir,
            symptoms_text=symptoms_text,
            api_key=api_key,
            selected_classes=cfg.get("classes"),
            images_per_class=cfg.get("images_per_class"),
        )
        all_results[cfg["dataset"]] = stats

    print("\n" + "=" * 60)
    print("GLOBAL SUMMARY")
    print("=" * 60)
    for ds, stats in all_results.items():
        print(f"  {ds}")
        print(f"    Accuracy       : {stats['accuracy']:.1f}%")
        print(f"    Avg tool calls : {stats['avg_tool_calls']:.1f}")
        print(f"    Avg ref images : {stats['avg_ref_turns']:.1f}")
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

    # ── Programmatic usage ────────────────────────────────────────────────────
    # run_agent(run_config=[
    #     {
    #         "dataset":          "Corn_Diseases",
    #         "classes":          ["Common_rust", "Gray_leaf_spot", "Tar_spot"],
    #         "images_per_class": 3,
    #     },
    #     {
    #         "dataset":          "Soybean_Diseases",
    #         "classes":          None,
    #         "images_per_class": 2,
    #     },
    # ])