"""Defensible Disease Registry Pipeline — CLI entry point.

Two independent pipelines:
  Local:    folder names + PDF → symptom extraction → local_registry.json
  Internet: folder names + web discovery → extraction → reconciliation → final_registry.json

Usage:
  python -m disease_registry.pipeline --crop soybean --track local \
    --pdf reference.pdf --disease-dir "Disease Images/"
  python -m disease_registry.pipeline --crop soybean --track internet \
    --disease-dir "Disease Images/"
  python -m disease_registry.pipeline --crop soybean --track both \
    --pdf reference.pdf --disease-dir "Disease Images/"
"""

import argparse
import json
import sys

from .local_pipeline import run_local_pipeline
from .internet_pipeline import run_internet_pipeline
from .shared import api_query
from .prompts import COMPARISON_PROMPT
from .utils import get_crop_dir, load_disease_names_from_dir, save_file


def run_pipeline(
    crop: str,
    track: str = "both",
    pdf_path: str | None = None,
    disease_dir: str | None = None,
    quick: bool = False,
    resume_from: str | None = None,
) -> dict:
    """Run local, internet, or both pipelines.

    Args:
        crop: Crop name (e.g., "soybean")
        track: Which pipeline to run — "local", "internet", or "both"
        pdf_path: Path to reference PDF (required for local track)
        disease_dir: Path to image directory with disease folders
        quick: Quick mode (fewer sources, shorter timeouts)
        resume_from: Resume internet pipeline from a specific stage
    """
    print(f"\n{'#'*60}")
    print(f"# DISEASE REGISTRY — {crop.upper()} (track: {track})")
    print(f"{'#'*60}")

    # Load disease names from image directory
    disease_names = None
    if disease_dir:
        disease_names = load_disease_names_from_dir(disease_dir, crop)
        print(f"  Loaded {len(disease_names)} known diseases from directory")

    local_registry = None
    internet_registry = None

    # ── Local Pipeline ──
    if track in ("local", "both"):
        if not pdf_path:
            print("ERROR: Local track requires --pdf")
            sys.exit(1)
        if not disease_names:
            print("ERROR: Local track requires --disease-dir to know which diseases to include")
            sys.exit(1)
        local_registry = run_local_pipeline(
            crop=crop,
            pdf_path=pdf_path,
            disease_names=disease_names,
        )

    # ── Internet Pipeline ──
    if track in ("internet", "both"):
        internet_registry = run_internet_pipeline(
            crop=crop,
            disease_names=disease_names,
            quick=quick,
            resume_from=resume_from,
        )

    # ── Cross-track Comparison ──
    if local_registry and internet_registry:
        print(f"\n{'='*60}")
        print(f"COMPARISON — Local vs Internet registries")
        print(f"{'='*60}")

        report = api_query(
            prompt=COMPARISON_PROMPT.format(
                registry=json.dumps(internet_registry, indent=2),
                expert_sheet=f"LOCAL REGISTRY (from PDF):\n{json.dumps(local_registry, indent=2)}",
            ),
            system_prompt=(
                "You are a QA agent for plant pathology data. "
                "Compare two disease registries — one from local sources (PDF), "
                "one from internet sources. Produce a detailed discrepancy report."
            ),
        )
        save_file("discrepancy_report.md", report or "", output_dir=get_crop_dir(crop))

    crop_dir = f"outputs/{crop.title()}/"
    print(f"\n{'#'*60}")
    print(f"# PIPELINE COMPLETE")
    if local_registry:
        print(f"#   Local:    {crop_dir}local_registry.json, {crop_dir}local.xlsx")
    if internet_registry:
        print(f"#   Internet: {crop_dir}final_registry.json, {crop_dir}internet.xlsx")
    print(f"{'#'*60}")

    return internet_registry or local_registry or {}


def main():
    parser = argparse.ArgumentParser(
        description="Generate a defensible, evidence-backed crop disease registry"
    )
    parser.add_argument("--crop", required=True, help="Crop name (e.g., soybean)")
    parser.add_argument("--track", default="both", choices=["local", "internet", "both"],
                        help="Which pipeline to run (default: both)")
    parser.add_argument("--pdf", default=None, help="Path to reference PDF (local track)")
    parser.add_argument("--disease-dir", default=None,
                        help="Path to image directory with disease folders")
    parser.add_argument("--quick", action="store_true", help="Quick mode for testing")
    parser.add_argument("--resume-from", default=None,
                        choices=["discovery", "extraction", "reconciliation"],
                        help="Resume internet pipeline from stage")
    args = parser.parse_args()

    run_pipeline(
        crop=args.crop,
        track=args.track,
        pdf_path=args.pdf,
        disease_dir=args.disease_dir,
        quick=args.quick,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
