#!/usr/bin/env python3
"""Analyze wheat disease evaluation results and generate comprehensive report."""

import json
from pathlib import Path
from collections import defaultdict
import statistics

def load_wheat_results():
    """Load all wheat evaluation results from summary files."""
    results_base = Path("results/open_agentic/Wheat_Diseases")
    results = {}

    for summary_file in results_base.glob("*/*/k*/summary.json"):
        parts = summary_file.parts
        src = parts[-4]
        model = parts[-3]
        k = int(parts[-2].replace("k", ""))

        with open(summary_file) as f:
            data = json.load(f)

        metrics = data.get("metrics", {})
        key = (model, src, k)
        results[key] = {
            "correct": metrics.get("correct", 0),
            "total": metrics.get("total", 0),
            "accuracy": metrics.get("accuracy", 0),
            "avg_refs": metrics.get("avg_refs_viewed", 0),
            "cost_per_img": metrics.get("avg_cost_usd", 0),
            "per_class": data.get("per_class_accuracy", {}),
            "avg_duration_s": metrics.get("avg_duration_s", 0),
        }

    return results

def analyze_k_sweep():
    """Analyze k-sweep results for each configuration."""
    results = load_wheat_results()

    print("=" * 80)
    print("WHEAT K-SWEEP ANALYSIS")
    print("=" * 80)
    print()

    # Group by model and source
    by_config = defaultdict(dict)
    for (model, src, k), data in results.items():
        key = (model, src)
        by_config[key][k] = data

    # Print k-sweep tables
    for (model, src) in sorted(by_config.keys()):
        print(f"\n{model.upper()} | {src.upper()}")
        print("-" * 60)
        print(f"{'K':>2} | {'Accuracy':>10} | {'Refs':>6} | {'Cost/img':>9} | {'Time/img':>9}")
        print("-" * 60)

        ks = sorted(by_config[(model, src)].keys())
        for k in ks:
            data = by_config[(model, src)][k]
            acc = data["accuracy"] * 100
            refs = data["avg_refs"]
            cost = data["cost_per_img"]
            time_s = data["avg_duration_s"]
            print(f"{k:2d} | {acc:9.1f}% | {refs:6.2f} | ${cost:8.5f} | {time_s:8.1f}s")

    print()

def analyze_model_ablation():
    """Analyze model ablation at k=8 internet KB."""
    results = load_wheat_results()

    print("=" * 80)
    print("MODEL ABLATION AT K=8 (INTERNET KB)")
    print("=" * 80)
    print()
    print(f"{'Model':>12} | {'Accuracy':>10} | {'Cost/img':>9} | {'Refs':>6}")
    print("-" * 60)

    for model in ["haiku", "sonnet", "opus", "gemini-flash", "gemini-pro"]:
        key = (model, "internet", 8)
        if key in results:
            data = results[key]
            acc = data["accuracy"] * 100
            cost = data["cost_per_img"]
            refs = data["avg_refs"]
            print(f"{model:>12} | {acc:9.1f}% | ${cost:8.5f} | {refs:6.2f}")
        else:
            print(f"{model:>12} | {'N/A':>10} | {'N/A':>9} | {'N/A':>6}")

    print()

def find_problem_classes():
    """Identify classes with lowest accuracy."""
    results = load_wheat_results()

    # Use internet KB at k=0 for analysis (most stable config)
    key = ("sonnet", "internet", 0)
    if key not in results:
        print("Warning: No sonnet/internet/k=0 results for per-class analysis")
        return

    per_class = results[key]["per_class"]

    print("=" * 80)
    print("PROBLEM CLASSES (SONNET, INTERNET KB, K=0)")
    print("=" * 80)
    print()

    # Find low performers
    low_performers = [(cls, acc) for cls, acc in per_class.items() if acc < 0.33]
    low_performers.sort(key=lambda x: x[1])

    if low_performers:
        print("Classes with <33% accuracy:")
        print(f"{'Disease Class':40} | {'Accuracy':>10}")
        print("-" * 55)
        for cls, acc in low_performers[:15]:
            print(f"{cls:40} | {acc*100:9.1f}%")

    print()

    # Find high performers
    high_performers = [(cls, acc) for cls, acc in per_class.items() if acc > 0.66]
    high_performers.sort(key=lambda x: x[1], reverse=True)

    if high_performers:
        print("\nClasses with >66% accuracy:")
        print(f"{'Disease Class':40} | {'Accuracy':>10}")
        print("-" * 55)
        for cls, acc in high_performers[:10]:
            print(f"{cls:40} | {acc*100:9.1f}%")

    print()

def main():
    """Generate complete analysis."""
    analyze_k_sweep()
    analyze_model_ablation()
    find_problem_classes()

    print("=" * 80)
    print("END OF ANALYSIS")
    print("=" * 80)

if __name__ == "__main__":
    main()
