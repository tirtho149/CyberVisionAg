"""KB Improver — Analyzes eval failures and rewrites KB descriptions.

Uses `claude -p` to analyze confusion patterns and reasoning traces from
eval results, then produces an improved KB markdown file with better
distinguishing features.

Usage:
    python -m CyberVisionAg.open_agentic.improve_kb \
        --kb-file CyberVisionAg/open_agentic/kb_v0.md \
        --results-dir CyberVisionAg/results/open_agentic/internet/logs/Soybean_Diseases \
        --output CyberVisionAg/open_agentic/kb_v1.md
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

ENV_STRIP_PREFIXES = ("CLAUDE", "CURSOR", "MCP_CONNECTION", "VSCODE", "ELECTRON")
TIMEOUT_S = 600  # 10 minutes — this is a heavy analysis task


def _clean_env() -> dict[str, str]:
    return {
        k: v for k, v in os.environ.items()
        if not any(k.startswith(prefix) for prefix in ENV_STRIP_PREFIXES)
    }


def load_eval_results(results_dir: Path) -> list[dict]:
    """Load all per-image result JSONs from the results directory."""
    results = []
    for f in sorted(results_dir.glob("*.json")):
        if f.name == "summary.json":
            continue
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def build_failure_analysis(results: list[dict]) -> str:
    """Build a concise failure analysis from eval results."""
    from collections import defaultdict

    # Per-class stats
    class_stats = defaultdict(lambda: {"correct": 0, "total": 0, "failures": []})
    for r in results:
        gt = r["ground_truth"]
        class_stats[gt]["total"] += 1
        if r["correct"]:
            class_stats[gt]["correct"] += 1
        elif not r.get("error"):
            class_stats[gt]["failures"].append({
                "predicted": r["prediction"],
                "confidence": r["confidence"],
                "reasoning": r.get("reasoning", "")[:500],
            })

    lines = ["## Evaluation Results & Failure Analysis\n"]

    # Summary
    total = sum(s["total"] for s in class_stats.values())
    correct = sum(s["correct"] for s in class_stats.values())
    lines.append(f"Overall: {correct}/{total} ({correct/total*100:.1f}%)\n")

    # Per-class with failure details
    for cls in sorted(class_stats.keys()):
        stats = class_stats[cls]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] else 0
        lines.append(f"### {cls}: {acc:.0f}% ({stats['correct']}/{stats['total']})")

        if stats["failures"]:
            lines.append("**Failures:**")
            for fail in stats["failures"]:
                lines.append(
                    f"- Predicted **{fail['predicted']}** (conf={fail['confidence']:.2f}): "
                    f"{fail['reasoning']}"
                )
        lines.append("")

    return "\n".join(lines)


def build_improver_prompt(current_kb: str, failure_analysis: str, only_failing: bool = True) -> str:
    """Build the prompt for the KB improver agent.

    Args:
        current_kb: Current KB markdown text.
        failure_analysis: Failure analysis text from eval results.
        only_failing: If True, only rewrite descriptions for classes that had
            misclassifications. Keeps working descriptions untouched to avoid
            the whack-a-mole problem.
    """
    if only_failing:
        rewrite_rule = (
            "- ONLY rewrite descriptions for classes that had failures (accuracy < 100%). "
            "Copy classes with 100% accuracy EXACTLY as-is — do NOT change working descriptions.\n"
        )
    else:
        rewrite_rule = (
            "- Rewrite ALL descriptions, including 100% classes — sharpen everything.\n"
        )

    return f"""You are a plant pathology expert improving a disease knowledge base (KB) used for image-based classification.

## Task

An AI agent uses the KB below to classify soybean disease images. The agent reads a test image, consults these descriptions, views reference images, and predicts a disease class. Below you'll find the current KB and a detailed failure analysis showing which classes were confused and the agent's reasoning for each wrong prediction.

Your job: **rewrite the KB descriptions to reduce misclassifications**. Focus on:

1. **Differential diagnosis**: For each failing disease, add what visually distinguishes it from the diseases it was confused WITH. Use phrases like "Unlike X, this disease shows..." or "Key difference from X: ..."
2. **Visual specificity**: Replace vague descriptions with concrete, image-observable features (colors, shapes, patterns, locations on plant)
3. **Common confusions**: Explicitly mention look-alikes and how to tell them apart
4. **Preserve accuracy**: Don't invent symptoms — only sharpen and reorganize what's already described

Rules:
{rewrite_rule}- Output the COMPLETE KB in the same markdown format (### DiseaseName followed by description)
- Include ALL diseases from the original KB
- Keep descriptions concise (2-4 sentences max per disease) — longer is not better
- Every disease MUST keep its exact original name (### header)
- Do NOT add new diseases or remove existing ones

## Current KB

{current_kb}

## Failure Analysis

{failure_analysis}

## Output

Write the complete KB below. Use the exact same format: ### DiseaseName followed by the description (improved for failing classes, copied verbatim for 100% classes).
"""


def run_improver(prompt: str) -> str:
    """Run claude -p with the improver prompt and return the improved KB."""
    cmd = [
        "claude", "-p",
        "--output-format", "text",
        "--model", "sonnet",
        "--verbose",
    ]

    env = _clean_env()

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(PROJECT_ROOT),
        start_new_session=True,
    )

    try:
        stdout_bytes, stderr_bytes = proc.communicate(
            input=prompt.encode(), timeout=TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
        sys.exit("ERROR: claude -p timed out")

    if proc.returncode != 0:
        stderr = stderr_bytes.decode(errors="replace")[:500]
        sys.exit(f"ERROR: claude -p failed (exit {proc.returncode}): {stderr}")

    return stdout_bytes.decode(errors="replace")


def extract_kb_from_response(response: str) -> str:
    """Extract the KB markdown from the agent's response.

    The response should contain ### headers with disease descriptions.
    """
    lines = response.strip().split("\n")
    kb_lines = []
    in_kb = False

    for line in lines:
        if line.startswith("### "):
            in_kb = True
        if in_kb:
            kb_lines.append(line)

    if not kb_lines:
        # Fallback: return the whole response
        return response.strip()

    return "\n".join(kb_lines).strip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Improve KB based on eval failures")
    parser.add_argument("--kb-file", required=True,
                        help="Current KB markdown file (e.g., kb_v0.md)")
    parser.add_argument("--results-dir", required=True,
                        help="Directory with per-image result JSONs")
    parser.add_argument("--output", required=True,
                        help="Output path for improved KB (e.g., kb_v1.md)")
    args = parser.parse_args()

    kb_path = Path(args.kb_file)
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)

    if not kb_path.exists():
        sys.exit(f"ERROR: KB file not found: {kb_path}")
    if not results_dir.exists():
        sys.exit(f"ERROR: Results directory not found: {results_dir}")

    # Load inputs
    current_kb = kb_path.read_text()
    results = load_eval_results(results_dir)
    print(f"Loaded {len(results)} eval results from {results_dir}")

    # Build failure analysis
    failure_analysis = build_failure_analysis(results)
    print(f"Failure analysis: {len(failure_analysis)} chars")

    # Build prompt
    prompt = build_improver_prompt(current_kb, failure_analysis)
    print(f"Prompt size: {len(prompt)} chars")
    print("Running KB improver agent (claude -p)...")

    # Run improver
    response = run_improver(prompt)

    # Extract KB
    improved_kb = extract_kb_from_response(response)

    # Validate: check that we got roughly the same number of diseases
    original_count = current_kb.count("### ")
    improved_count = improved_kb.count("### ")
    print(f"Diseases: {original_count} original → {improved_count} improved")

    if improved_count < original_count * 0.8:
        print(f"WARNING: Improved KB has fewer diseases ({improved_count} vs {original_count})")

    # Save
    output_path.write_text(improved_kb)
    print(f"Saved improved KB: {output_path}")


if __name__ == "__main__":
    main()
