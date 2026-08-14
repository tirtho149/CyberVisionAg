"""Local Pipeline — Build disease registry from folder names + reference PDF.

Inputs: disease name list (from image folders), reference PDF
Process: PDF page-by-page extraction → match symptoms to folder diseases → output
Output: local_registry.json, local_registry.md, {Crop}_local.xlsx
"""

import base64
import concurrent.futures
import time
from pathlib import Path

from .config import (
    MAX_PARALLEL_EXTRACTIONS,
    PDF_PAGES_PER_CHUNK,
    PDF_EXTRACTION_MAX_TOKENS,
)
from .prompts import EXTRACTION_SCHEMA, PDF_PAGE_EXTRACTION_PROMPT
from .shared import api_query, match_names_to_folders, parse_json_result
from .utils import (
    get_crop_dir,
    registry_to_markdown,
    save_file,
    save_json,
    today_iso,
    write_enriched_xlsx,
)


# ─── PDF Extraction ─────────────────────────────────────────────────────────


def _extract_pdf_chunk(pdf_bytes: bytes, page_start: int, page_end: int,
                       crop: str, disease_names: list[str], pdf_name: str,
                       chunk_label: str) -> list[dict]:
    """Extract diseases from a chunk of PDF pages via Anthropic API."""
    print(f"  {chunk_label}", flush=True)

    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    content_blocks = [{
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": pdf_b64,
        },
        "cache_control": {"type": "ephemeral"},
    }]

    disease_list = "\n".join(f"- {name.replace('_', ' ')}" for name in disease_names)
    prompt = PDF_PAGE_EXTRACTION_PROMPT.format(
        crop=crop, disease_list=disease_list, pdf_name=pdf_name
    )

    raw = api_query(
        prompt=prompt,
        system_prompt=(
            "You are a data extraction agent for plant disease information. "
            "Extract disease data from the provided PDF with verbatim quotes. "
            "NEVER fill in fields from your own knowledge. "
            "Output ONLY valid JSON matching the required schema."
        ),
        json_schema=EXTRACTION_SCHEMA,
        content_blocks=content_blocks,
        max_tokens=PDF_EXTRACTION_MAX_TOKENS,
    )

    result = parse_json_result(raw, f"pdf_extraction_p{page_start}-{page_end}")
    extractions = result.get("extractions", [])

    # Tag all extractions with PDF source info
    for ext in extractions:
        ext["source_url"] = f"pdf://{pdf_name}"
        ext["source_title"] = f"PDF: {pdf_name} (pages {page_start}-{page_end})"
        ext["source_type"] = "pdf_reference"
        ext["access_date"] = today_iso()

    n_diseases = sum(len(e.get("extracted_diseases", [])) for e in extractions)
    print(f"    → {n_diseases} diseases from pages {page_start}-{page_end}")
    return extractions


def _run_pdf_extraction(pdf_path: str, crop: str, disease_names: list[str]) -> dict:
    """Extract disease data from a PDF, page-by-page in parallel chunks."""
    print(f"\n{'='*60}")
    print(f"PDF EXTRACTION — Reading {Path(pdf_path).name} page by page")
    print(f"{'='*60}")
    t0 = time.time()

    pdf_name = Path(pdf_path).name
    pdf_bytes = Path(pdf_path).read_bytes()

    # Get page count using a lightweight PDF reader
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
    except ImportError:
        # Fallback: send the whole PDF as one chunk
        print("  WARNING: PyMuPDF not installed, sending entire PDF as one chunk")
        all_extractions = _extract_pdf_chunk(
            pdf_bytes, 1, 0, crop, disease_names, pdf_name, "Chunk 1/1: entire PDF"
        )
        data = {"extractions": all_extractions}
        n_total = sum(len(e.get("extracted_diseases", [])) for e in all_extractions)
        print(f"\n  PDF TOTAL: {n_total} disease records ({time.time()-t0:.0f}s)")
        return data

    print(f"  PDF has {total_pages} pages, processing in chunks of {PDF_PAGES_PER_CHUNK}")

    # Pre-split PDF into chunk bytes (open doc once)
    doc = fitz.open(pdf_path)
    n_chunks = (total_pages + PDF_PAGES_PER_CHUNK - 1) // PDF_PAGES_PER_CHUNK
    chunk_args = []
    for i, start in enumerate(range(0, total_pages, PDF_PAGES_PER_CHUNK)):
        end = min(start + PDF_PAGES_PER_CHUNK, total_pages)
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(doc, from_page=start, to_page=end - 1)
        chunk_bytes = chunk_doc.tobytes()
        chunk_doc.close()
        label = f"Chunk {i+1}/{n_chunks}: pages {start+1}-{end}"
        chunk_args.append((chunk_bytes, start + 1, end, crop, disease_names, pdf_name, label))
    doc.close()

    # Extract all chunks in parallel
    all_extractions = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_EXTRACTIONS) as executor:
        futures = [executor.submit(_extract_pdf_chunk, *args) for args in chunk_args]
        for future in futures:
            all_extractions.extend(future.result())

    data = {"extractions": all_extractions}
    n_total = sum(len(e.get("extracted_diseases", [])) for e in all_extractions)
    print(f"\n  PDF TOTAL: {n_total} disease records ({time.time()-t0:.0f}s)")
    return data


# ─── Local Registry Building ────────────────────────────────────────────────


def _build_local_registry(pdf_extractions: dict, crop: str,
                          disease_names: list[str]) -> dict:
    """Build local registry from folder names + PDF symptoms.

    Disease list comes from folder names (disease_names). The PDF provides
    visual symptoms via extraction. LLM-based matching maps PDF disease names
    to folder names.
    """
    # Collect all PDF-extracted diseases into a lookup by exact extracted name
    pdf_diseases: dict[str, tuple[dict, str]] = {}
    pdf_names = []
    for source in pdf_extractions.get("extractions", []):
        source_url = source.get("source_url", "")
        for disease in source.get("extracted_diseases", []):
            name_field = disease.get("disease_name", {})
            name = name_field.get("value") if isinstance(name_field, dict) else name_field
            if name:
                pdf_diseases[name] = (disease, source_url)
                pdf_names.append(name)

    # LLM-based matching: PDF names → folder names
    print(f"  Matching {len(pdf_names)} PDF diseases to {len(disease_names)} folder names...")
    name_map = match_names_to_folders(pdf_names, disease_names)
    # Invert: folder_name → (pdf_disease, pdf_url) using best match
    folder_to_pdf: dict[str, tuple[dict, str]] = {}
    for pdf_name, folder_name in name_map.items():
        key = folder_name.strip().lower()
        if key not in folder_to_pdf and pdf_name in pdf_diseases:
            folder_to_pdf[key] = pdf_diseases[pdf_name]

    def _to_cited(field, url=""):
        """Convert extraction evidence field to registry cited field."""
        if not field or not isinstance(field, dict):
            return {"value": None, "url": None, "quote": None}
        val = field.get("value")
        evidence = field.get("evidence")
        return {"value": val, "url": url or None, "quote": evidence}

    _null_cited = {"value": None, "url": None, "quote": None}

    # Build registry: iterate over folder names (the canonical disease list)
    registry_diseases = []
    matched_pdf = 0
    for name in disease_names:
        key = name.strip().lower()

        # Look up PDF extraction for this disease
        pdf_result = folder_to_pdf.get(key)
        pdf_entry, pdf_url = pdf_result if pdf_result else (None, "")

        entry = {
            "disease_name": name,
            "pathogen_scientific_name": dict(_null_cited),
            "type_of_disease": dict(_null_cited),
            "affected_parts": dict(_null_cited),
            "visual_symptoms": {
                "summary": dict(_null_cited),
                "diagnostic_features": dict(_null_cited),
                "look_alikes": dict(_null_cited),
            },
            "confidence": "low",
            "num_sources": 0,
        }

        # Fill from PDF if matched
        if pdf_entry:
            matched_pdf += 1
            entry["confidence"] = "medium"
            entry["num_sources"] = 1

            # Affected parts from PDF
            entry["affected_parts"] = _to_cited(pdf_entry.get("affected_parts", {}), pdf_url)

            # Pathogen and type from PDF (if available)
            entry["pathogen_scientific_name"] = _to_cited(
                pdf_entry.get("pathogen_scientific_name", {}), pdf_url)
            entry["type_of_disease"] = _to_cited(
                pdf_entry.get("type_of_disease", {}), pdf_url)

            # Visual symptoms
            vs = pdf_entry.get("visual_symptoms", {})
            if isinstance(vs, dict):
                entry["visual_symptoms"] = {
                    "summary": _to_cited(vs.get("summary", {}), pdf_url),
                    "diagnostic_features": _to_cited(vs.get("diagnostic_features", {}), pdf_url),
                    "look_alikes": _to_cited(vs.get("look_alikes", {}), pdf_url),
                }

        registry_diseases.append(entry)

    print(f"  Local registry: {len(registry_diseases)} diseases, {matched_pdf} matched with PDF symptoms")

    return {
        "diseases": registry_diseases,
        "crop": crop,
        "generated_date": today_iso(),
    }


# ─── Pipeline Orchestrator ──────────────────────────────────────────────────


def run_local_pipeline(
    crop: str,
    pdf_path: str,
    disease_names: list[str],
) -> dict:
    """Run the local pipeline: folder names + PDF extraction → local registry.

    Args:
        crop: Crop name (e.g., "soybean")
        pdf_path: Path to reference PDF
        disease_names: List of known disease names (from image directory)

    Returns:
        Local registry dict with diseases, crop, generated_date
    """
    print(f"\n{'='*60}")
    print(f"LOCAL PIPELINE — {crop.upper()}")
    print(f"{'='*60}")

    output_dir = get_crop_dir(crop)

    # Step 1: Extract symptoms from PDF
    pdf_data = _run_pdf_extraction(pdf_path, crop, disease_names)
    save_json("pdf_extractions.json", pdf_data, output_dir=output_dir)

    # Step 2: Build registry from folder names + PDF symptoms
    registry = _build_local_registry(pdf_data, crop, disease_names)
    save_json("local_registry.json", registry, output_dir=output_dir)
    save_file("local_registry.md", registry_to_markdown(registry), output_dir=output_dir)

    # Step 3: Write xlsx
    xlsx_path = str(output_dir / "local.xlsx")
    write_enriched_xlsx(registry, None, xlsx_path)

    return registry
