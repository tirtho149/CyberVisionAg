#!/usr/bin/env python3
"""
generate_symptoms.py
====================
Reads the Curated_Local_Dataset produced by dataloader.py and generates
a single structured symptom knowledge-base markdown file, crop-wise.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW IT WORKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — Dataset discovery
  Scans Curated_Local_Dataset/train/<Category>/<Disease>/
  to build the complete crop → disease map automatically.

STEP 2 — Document upload (Files API)
  For each file found in knowledge_docs/ the script uploads it ONCE to
  OpenAI's Files API (purpose="user_data") and caches the returned
  file_id for the session.

  Accepted reference document formats (all handled natively by OpenAI):
    .pdf   → text + page images extracted by OpenAI (best fidelity)
    .docx  → text extracted
    .pptx  → text extracted
    .txt / .md / .rst / .html / .json → plain text
    .csv / .xlsx / .tsv → spreadsheet augmentation (first 1 000 rows)

  No local PDF parsing library required.  The model sees the full
  document including embedded diagrams (for PDFs).

STEP 3 — Symptom extraction / generation (Responses API)
  For every disease the script calls the Responses API:
    • If a matching reference doc was uploaded → the file_id is passed as
      an input_file item alongside the extraction prompt.
      OpenAI handles all parsing internally.
    • If no doc exists → a generation-only prompt is sent.

  Both paths use the same Responses API endpoint (/v1/responses).

STEP 4 — Output
  Writes one combined  disease_symptoms_crop_wise.md:
    ## Crop: Soybean Diseases
    ### Anthracnose
    **Symptoms:** …
    **Reference Images:** …
    ---
    ## Crop: Corn Diseases
    …

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIRECTORY LAYOUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
project/
  Curated_Local_Dataset/          ← output of dataloader.py
    train/
      Soybean_Diseases/
        Anthracnose/
          Anthracnose_train_001.jpg …
      Corn_Diseases/ …
    test/ …
    metadata.json

  knowledge_docs/                  ← ONE reference doc per crop (any format)
    soybean.pdf                    ← PDF  → text + images (best)
    corn.docx                      ← Word doc → text
    mango.md                       ← Markdown → text
    alfalfa.txt                    ← Plain text → text
    wheat.xlsx                     ← Spreadsheet → augmented
    …

  generate_symptoms.py             ← THIS FILE
  disease_symptoms_crop_wise.md    ← single combined output

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python generate_symptoms.py [OPTIONS]

  --curated   PATH   Curated_Local_Dataset dir   (default: ./Curated_Local_Dataset)
  --docs      PATH   knowledge_docs dir           (default: ./knowledge_docs)
  --output    PATH   output markdown file         (default: ./disease_symptoms_crop_wise.md)
  --model     MODEL  OpenAI model to use          (default: gpt-4o)
  --split     SPLIT  train | test                 (default: train)
  --no-delete        Keep uploaded files on OpenAI after the run
  --dry-run          Print discovered structure, no API calls

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install openai python-dotenv
"""

from __future__ import annotations

import os
import re
import sys
import argparse
import textwrap
import mimetypes
from pathlib import Path
from dotenv import dotenv_values
from openai import OpenAI

# ── load .env ──────────────────────────────────────────────────────────────────
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for k, v in dotenv_values(_env_file).items():
        if v:
            os.environ[k] = v

# ── constants ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL  = "gpt-4o-mini"
MAX_TOKENS     = 600
MAX_REF_IMAGES = 3
IMAGE_EXT      = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

# All file types the OpenAI Files API / Responses API accepts as input_file.
# Maps extension → MIME type override (mimetypes module often gets these wrong).
ACCEPTED_DOC_TYPES = {
    # PDFs — text + page images (best fidelity)
    ".pdf":  "application/pdf",
    # Rich documents
    ".doc":  "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".odt":  "application/vnd.oasis.opendocument.text",
    ".rtf":  "application/rtf",
    # Presentations
    ".ppt":  "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    # Spreadsheets (spreadsheet augmentation — first 1 000 rows)
    ".xls":  "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv":  "text/csv",
    ".tsv":  "text/tsv",
    # Text / code / markup
    ".txt":  "text/plain",
    ".md":   "text/markdown",
    ".rst":  "text/x-rst",
    ".html": "text/html",
    ".htm":  "text/html",
    ".json": "application/json",
    ".xml":  "text/xml",
}

# ── system prompts ─────────────────────────────────────────────────────────────
_SYSTEM_EXTRACT = textwrap.dedent("""
    You are a plant pathology expert and a retriever-first diagnostic writer.
    A reference document is attached. Your job is to extract ONLY the visual
    symptom information for a specific plant disease from that document.

    STRICT RULES:
    - Use ONLY information present in the attached document. Do NOT invent symptoms.
    - Focus exclusively on observable visual features:
        color, shape, texture, size, elevation (raised / sunken / flat),
        surface relation (superficial / embedded), halos, margins,
        distribution across the plant, and progression over time.
    - Ignore treatment, pathogen biology, management, and epidemiology.
    - End with one KEY DIFFERENCE sentence that compares this disease to the
      most visually similar disease on the same crop, if such comparison
      appears in the document.
    - Write as ONE coherent paragraph (5-8 sentences). No bullet points.
    - If the document contains NO visual symptom information for this specific
      disease, reply with exactly the single token:  NO_INFO
""").strip()

_SYSTEM_GENERATE = textwrap.dedent("""
    You are a plant pathology expert specialising in visual disease
    classification for machine learning and field scouting.
    Write accurate, observable, visual-first symptom descriptions.
""").strip()

_PROMPT_GENERATE = textwrap.dedent("""
    Write a 5-8 sentence visual symptom description for "{disease}" on {crop}.

    Cover ALL of the following:
    - Distinctive visual patterns: spots, lesions, galls, coatings,
      cankers, wilting, discolouration
    - Colour and texture of affected tissue
    - Shape, size, and distribution of symptoms across the plant
    - Whether symptoms are raised, sunken, or superficial
    - Presence of halos, margins, or associated structures
      (spores, exudate, mycelium, sclerotia, gummosis)
    - Which plant parts are affected (leaf, stem, root, pod, seed, etc.)
    - End with one KEY DIFFERENCE sentence comparing this disease
      to the most visually similar disease on the same crop.

    Write as ONE coherent paragraph. No bullet points.
    Be specific about features a machine-learning vision model would use
    to distinguish this disease from others.
""").strip()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Discover structure from Curated_Local_Dataset
# ══════════════════════════════════════════════════════════════════════════════

def discover_curated_dataset(curated_dir: Path, split: str = "train") -> dict:
    """
    Walk <curated_dir>/<split>/<Category>/<Disease>/ and return:
        { "Soybean_Diseases": { "Anthracnose": ["rel/img1.jpg", ...], ... }, ... }
    Image paths are relative to curated_dir.
    """
    split_dir = curated_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}\n"
            "Have you run dataloader.py first?"
        )

    structure = {}
    for category_dir in sorted(split_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        structure[category_dir.name] = {}
        for disease_dir in sorted(category_dir.iterdir()):
            if not disease_dir.is_dir():
                continue
            images = sorted(
                str(f.relative_to(curated_dir))
                for f in disease_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXT
            )
            structure[category_dir.name][disease_dir.name] = images[:MAX_REF_IMAGES]

    return structure


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Upload knowledge documents to OpenAI Files API
# ══════════════════════════════════════════════════════════════════════════════

def upload_knowledge_docs(client: OpenAI, docs_dir: Path):
    """
    Scan docs_dir and upload EVERY supported file to the OpenAI Files API.

    Returns:
        file_registry : list of dicts, one per uploaded file:
                        { "file_id": str, "path": Path, "stem_norm": str }
        file_paths    : { file_id: Path }  — flat map for deletion

    No file is ever skipped or overwritten regardless of name or format.
    The full original filename is preserved; matching is done at query time.
    """
    file_registry = []
    file_paths    = {}

    if not docs_dir.exists():
        print(f"  [INFO] knowledge_docs folder not found: {docs_dir}")
        return file_registry, file_paths

    docs = sorted(
        f for f in docs_dir.iterdir()
        if f.is_file() and f.suffix.lower() in ACCEPTED_DOC_TYPES
    )

    if not docs:
        print(f"  [INFO] No supported documents found in {docs_dir}")
        return file_registry, file_paths

    for doc in docs:
        ext  = doc.suffix.lower()
        mime = ACCEPTED_DOC_TYPES[ext]
        print(f"  [UPLOAD] {doc.name:<50} → ", end="", flush=True)
        try:
            with open(doc, "rb") as fh:
                response = client.files.create(
                    file=(doc.name, fh, mime),
                    purpose="user_data",
                )
            fid = response.id
            file_registry.append({
                "file_id":   fid,
                "path":      doc,
                "stem_norm": _normalise(doc.stem),   # used for crop matching
            })
            file_paths[fid] = doc
            print(f"file_id={fid}  [{ext}]")
        except Exception as exc:
            print(f"FAILED ({exc})")

    return file_registry, file_paths


def delete_uploaded_files(client: OpenAI, file_paths: dict) -> None:
    """Delete all files uploaded during this run to avoid storage accumulation."""
    for fid, path in file_paths.items():
        try:
            client.files.delete(fid)
            print(f"  [DELETE] {fid}  ({path.name})")
        except Exception as exc:
            print(f"  [WARN] Could not delete {fid}: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Fuzzy crop ↔ document matching
# ══════════════════════════════════════════════════════════════════════════════

def _normalise(name: str) -> str:
    """Lower-case, strip underscores/hyphens/spaces."""
    return re.sub(r"[\s_\-]+", "", name.lower())


def find_file_ids_for_category(category: str, file_registry: list) -> list:
    """
    Given a category like 'Soybean_Diseases', return the file_ids of ALL
    uploaded documents that mention any word from that category name.

    Matching: each word of the category is checked against the normalised
    stem of every uploaded file.  A file matches if ANY crop word appears
    anywhere inside its stem.

    Examples:
      'Soybean_Diseases'  matches  'soybeanDiseasesInIOWA...'  (word 'soybean')
      'Mango_Leaf_Disease' matches  'mango'                    (word 'mango')
      'Corn_Diseases'      matches  'cornDiseasesGuide'        (word 'corn')
    """
    crop_words = [
        w for w in re.split(r"[_\s]+", category.lower())
        if len(w) > 3  # skip short stop-words like 'and', 'in', 'of'
    ]

    matched = []
    for entry in file_registry:
        stem = entry["stem_norm"]
        if any(word in stem for word in crop_words):
            matched.append(entry["file_id"])

    return matched


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Symptom extraction / generation via Responses API
# ══════════════════════════════════════════════════════════════════════════════

def call_responses_api(client: OpenAI, model: str, system: str,
                       user_content: list) -> str:
    """
    Call the Responses API (/v1/responses) with arbitrary content blocks.
    Returns the text of the first output message.
    """
    response = client.responses.create(
        model=model,
        instructions=system,
        input=[
            {
                "role": "user",
                "content": user_content,
            }
        ],
        max_output_tokens=MAX_TOKENS,
    )
    # Extract text from all output content blocks
    text_parts = []
    for item in response.output:
        if hasattr(item, "content"):
            for block in item.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
    return "\n".join(text_parts).strip()


def extract_from_files(client: OpenAI, model: str,
                       crop: str, disease: str, file_ids: list) -> str:
    """
    Use the Responses API with one or more input_file items to extract symptoms
    grounded in the uploaded reference documents.
    All matched files (e.g. PDF + xlsx) are attached so the model sees both.
    """
    clean_disease = disease.replace("_", " ")
    clean_crop    = crop.replace("_", " ")

    # Attach every matched file first (PDF first due to upload sort order)
    user_content = [
        {"type": "input_file", "file_id": fid}
        for fid in file_ids
    ]

    # Then the extraction instruction
    user_content.append({
        "type": "input_text",
        "text": (
            f"Crop: {clean_crop}\n"
            f"Disease: {clean_disease}\n\n"
            f"Using the attached reference document(s), extract the visual "
            f"symptom description for '{clean_disease}' on {clean_crop}.\n\n"
            f"Look carefully — the disease may appear under a slightly different "
            f"name, as part of a group, or in a table. If you find any related "
            f"visual information, use it. Only reply NO_INFO if the disease is "
            f"truly absent from all attached documents."
        ),
    })

    result = call_responses_api(client, model, _SYSTEM_EXTRACT, user_content)
    return "" if result.strip() == "NO_INFO" else result


def generate_symptom(client: OpenAI, model: str,
                     crop: str, disease: str) -> str:
    """
    Use the Responses API (text-only) to generate symptoms from GPT knowledge
    when no reference document is available.
    """
    clean_disease = disease.replace("_", " ")
    clean_crop    = crop.replace("_", " ")

    user_content = [
        {
            "type": "input_text",
            "text": _PROMPT_GENERATE.format(
                disease=clean_disease,
                crop=clean_crop,
            ),
        }
    ]

    return call_responses_api(client, model, _SYSTEM_GENERATE, user_content)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Render combined markdown output
# ══════════════════════════════════════════════════════════════════════════════

def render_crop_section(category: str, diseases: dict,
                        symptoms_map: dict, curated_dir_name: str) -> str:
    """Render one ## Crop section with all its ### Disease blocks."""
    crop_display = category.replace("_", " ").title()
    parts = [f"## Crop: {crop_display}\n\n"]

    for disease, images in diseases.items():
        disease_display = disease.replace("_", " ").title()
        symptoms = symptoms_map.get(disease, "*(symptoms not available)*")

        image_lines = (
            "\n".join(f"- {curated_dir_name}/{img}" for img in images)
            if images else "- *(no reference images found)*"
        )

        parts.append(
            f"### {disease_display}\n\n"
            f"**Symptoms:**\n{symptoms}\n\n"
            f"**Reference Images:**\n{image_lines}\n\n"
            f"---\n\n"
        )

    return "".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a crop-wise plant disease symptom knowledge base."
    )
    p.add_argument(
        "--curated",
        default=str(Path(__file__).parent / "Curated_Local_Dataset"),
        help="Path to Curated_Local_Dataset (output of dataloader.py).",
    )
    p.add_argument(
        "--docs",
        default=str(Path(__file__).parent / "knowledge_docs"),
        help=(
            "Folder of reference documents — one per crop.  "
            "Accepts: .pdf .docx .pptx .doc .odt .rtf .xls .xlsx .csv .tsv "
            ".txt .md .rst .html .json .xml"
        ),
    )
    p.add_argument(
        "--output",
        default=str(Path(__file__).parent / "disease_symptoms_crop_wise.md"),
        help="Output combined markdown file.",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Dataset split to scan for disease folders (default: train).",
    )
    p.add_argument(
        "--no-delete",
        action="store_true",
        help="Do NOT delete uploaded files from OpenAI after the run.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover structure and show what would be uploaded; no API calls.",
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args    = parse_args()
    curated = Path(args.curated)
    docs    = Path(args.docs)
    out     = Path(args.output)
    model   = args.model

    print("=" * 65)
    print("  DISEASE SYMPTOM KNOWLEDGE BASE GENERATOR")
    print("=" * 65)
    print(f"  Curated dataset : {curated}")
    print(f"  Knowledge docs  : {docs}")
    print(f"  Output file     : {out}")
    print(f"  Model           : {model}")
    print(f"  Split           : {args.split}")
    print(f"  Delete uploads  : {not args.no_delete}")
    print("=" * 65)

    # ── Step 1: discover dataset structure ────────────────────────────────
    print(f"\n[1/3] Scanning {curated / args.split} ...\n")
    try:
        structure = discover_curated_dataset(curated, args.split)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    if not structure:
        print("[ERROR] No crop categories found. Check --curated path.")
        sys.exit(1)

    total_diseases = sum(len(d) for d in structure.values())
    print(f"  Found {len(structure)} crop categor(y/ies), "
          f"{total_diseases} disease class(es):\n")
    for cat, diseases in structure.items():
        print(f"    {cat:<42}  {len(diseases):3d} disease(s)")

    # Show what docs would be uploaded
    if docs.exists():
        candidates = [
            f for f in sorted(docs.iterdir())
            if f.suffix.lower() in ACCEPTED_DOC_TYPES
        ]
        print(f"\n  Knowledge docs found in {docs}:")
        for f in candidates:
            print(f"    {f.name}")
        if not candidates:
            print("    (none)")
    else:
        print(f"\n  Knowledge docs folder not found ({docs}) — will generate all.")

    if args.dry_run:
        print("\n[DRY RUN] Stopping before any API calls.")
        return

    # ── Initialise OpenAI client ───────────────────────────────────────────
    if not os.environ.get("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)
    client = OpenAI()

    # ── Step 2: upload all knowledge docs once ────────────────────────────
    print(f"\n[2/3] Uploading knowledge docs to OpenAI Files API ...\n")
    file_registry, file_paths = upload_knowledge_docs(client, docs)

    if file_registry:
        print(f"\n  {len(file_registry)} file(s) uploaded successfully.")
        for entry in file_registry:
            print(f"    {entry['path'].name:<50}  {entry['file_id']}")
    else:
        print("  No docs uploaded — all symptoms will be generated by GPT.")

    # ── Step 3 + 4: generate symptoms → build combined MD ─────────────────
    print(f"\n[3/3] Generating symptoms using Responses API ...\n")

    md_parts = [
        "# Plant Disease Knowledge Base (Crop-Wise)\n\n",
        "Visual symptom descriptions for plant disease classification, "
        "organised by crop.\n",
        f"Auto-generated from `{curated.name}/` using `generate_symptoms.py`.\n",
        "Symptoms are **extracted** from reference documents via the OpenAI "
        "Files API where available, otherwise **generated** by the model.\n\n",
        "---\n\n",
    ]

    for category, diseases in structure.items():
        crop_display = category.replace("_", " ").title()
        print(f"{'─'*65}")
        print(f"  {crop_display}  ({len(diseases)} diseases)")
        print(f"{'─'*65}")

        file_ids_for_cat = find_file_ids_for_category(category, file_registry)
        if file_ids_for_cat:
            mode = f"EXTRACTED  [{len(file_ids_for_cat)} file(s) attached]"
        else:
            mode = "GENERATED (no doc)"
        print(f"  Mode: {mode}\n")

        symptoms_map = {}

        for disease, images in diseases.items():
            print(f"    • {disease:<52}", end="", flush=True)

            if file_ids_for_cat:
                symptoms = extract_from_files(client, model, category, disease, file_ids_for_cat)
                if not symptoms:
                    print("(not in doc → generating) ", end="", flush=True)
                    symptoms = generate_symptom(client, model, category, disease)
            else:
                symptoms = generate_symptom(client, model, category, disease)

            symptoms_map[disease] = symptoms
            print("✓")

        md_parts.append(
            render_crop_section(category, diseases, symptoms_map, curated.name)
        )
        print()

    # ── Write output ───────────────────────────────────────────────────────
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(md_parts), encoding="utf-8")

    print(f"{'='*65}")
    print(f"  ✅ Knowledge base saved to: {out.resolve()}")
    print(f"{'='*65}\n")

    # ── Cleanup: delete uploaded files unless --no-delete ─────────────────
    if file_paths and not args.no_delete:
        print("Cleaning up uploaded files from OpenAI ...\n")
        delete_uploaded_files(client, file_paths)
        print()


if __name__ == "__main__":
    main()