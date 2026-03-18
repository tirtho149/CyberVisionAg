"""Build an attractor guide from eval results.

Reads per-image result JSONs, finds "attractor" classes (ones that get
over-predicted), and generates a markdown guide the agent reads AFTER forming
its initial prediction.

The guide is keyed by predicted class. For each attractor, it lists what the
actual class usually was when that prediction turned out wrong.

Usage:
    python -m CyberVisionAg.open_agentic.build_confusion_guide \
        --results-dirs \
            CyberVisionAg/results-mar16/open_agentic/Soybean_Diseases/none/sonnet/k8 \
            CyberVisionAg/results-mar16/open_agentic/Soybean_Diseases/internet/sonnet/k8 \
        --output CyberVisionAg/open_agentic/confusion_guides/Soybean_Diseases.md
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_results(results_dir: Path) -> list[dict]:
    """Load per-image result JSONs from a directory."""
    results = []
    for f in sorted(results_dir.glob("*.json")):
        if f.name == "summary.json":
            continue
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def build_attractor_map(
    all_results: list[dict],
    top_n: int = 4,
) -> dict[str, list[tuple[str, int]]]:
    """Build attractor map: {predicted_class: [(actual_class, count), ...]}.

    Takes the top_n most over-predicted classes by total wrong count.
    """
    # Count: when class X was predicted, what was the actual class?
    wrong_by_pred = defaultdict(lambda: defaultdict(int))
    for r in all_results:
        gt = r.get("ground_truth", "")
        pred = r.get("prediction", "")
        if not r.get("correct") and not r.get("error") and gt and pred:
            wrong_by_pred[pred][gt] += 1

    # Rank all classes by total wrong predictions, take top_n
    ranked = sorted(
        wrong_by_pred.items(),
        key=lambda x: -sum(x[1].values()),
    )
    attractors = {}
    for pred_cls, actuals in ranked[:top_n]:
        sorted_actuals = sorted(actuals.items(), key=lambda x: -x[1])
        total = sum(v for _, v in sorted_actuals)
        attractors[pred_cls] = sorted_actuals

    return attractors


def generate_attractor_guide(
    attractors: dict[str, list[tuple[str, int]]],
) -> str:
    """Generate the attractor guide markdown."""
    lines = [
        "# Attractor Guide",
        "",
        "Some classes get over-predicted. If your initial prediction is one of",
        "the classes listed below, consider the alternatives before confirming.",
        "",
    ]

    for pred_cls in sorted(attractors, key=lambda c: -sum(n for _, n in attractors[c])):
        actuals = attractors[pred_cls]
        total = sum(n for _, n in actuals)
        lines.append(f"## {pred_cls}")
        lines.append(f"This class is over-predicted ({total}x wrong in past evaluations).")
        lines.append("When agents predicted this class, the actual class was:")
        for actual, count in actuals:
            lines.append(f"- {actual} ({count}x)")
        lines.append("Before confirming, view references for these alternatives.")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Build attractor guide from eval results"
    )
    parser.add_argument(
        "--results-dirs", nargs="+", required=True,
        help="One or more result directories containing per-image JSONs",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output markdown file path",
    )
    parser.add_argument(
        "--top-n", type=int, default=4,
        help="Number of top attractor classes to include (default: 4)",
    )
    args = parser.parse_args()

    # Load all results
    all_results = []
    for d in args.results_dirs:
        path = Path(d)
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        results = load_results(path)
        print(f"Loaded {len(results)} results from {path}")
        all_results.extend(results)

    if not all_results:
        print("ERROR: No results found")
        return

    # Build attractor map
    attractors = build_attractor_map(all_results, top_n=args.top_n)

    print(f"\nTotal results: {len(all_results)}")
    print(f"Top {args.top_n} attractor classes:")
    for pred_cls in sorted(attractors, key=lambda c: -sum(n for _, n in attractors[c])):
        total = sum(n for _, n in attractors[pred_cls])
        actuals_str = ", ".join(f"{a}({n})" for a, n in attractors[pred_cls])
        print(f"  {pred_cls}: {total}x wrong -> {actuals_str}")

    # Save per-class files
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Clear old files
    for old in output_dir.glob("*.md"):
        old.unlink()
    for pred_cls in attractors:
        actuals = attractors[pred_cls]
        total = sum(n for _, n in actuals)
        lines = [
            f"This class is over-predicted ({total}x wrong in past evaluations).",
            "When agents predicted this class, the actual class was:",
        ]
        for actual, count in actuals:
            lines.append(f"- {actual} ({count}x)")
        lines.append("Before confirming, view references for these alternatives.")
        (output_dir / f"{pred_cls}.md").write_text("\n".join(lines))
    print(f"\nSaved {len(attractors)} attractor files to: {output_dir}/")


if __name__ == "__main__":
    main()
