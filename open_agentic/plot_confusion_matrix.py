"""Generate confusion matrix heatmap from eval results."""
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_confusion_matrix(result_dir: str, output_path: str, title: str = ""):
    result_dir = Path(result_dir)
    all_results = []
    for f in sorted(result_dir.glob('*.json')):
        if f.name == 'summary.json':
            continue
        all_results.append(json.load(open(f)))

    classes = sorted(set(r['ground_truth'] for r in all_results))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    matrix = np.zeros((n, n), dtype=int)

    for r in all_results:
        gt = r['ground_truth']
        pred = r.get('prediction', '')
        if pred in class_to_idx:
            matrix[class_to_idx[gt], class_to_idx[pred]] += 1

    correct = int(sum(matrix[i, i] for i in range(n)))
    print(f"Classes: {n}, Total: {len(all_results)}, Correct: {correct}/{len(all_results)} ({correct/len(all_results)*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(18, 16))

    # Shorten class names for readability
    short = []
    for c in classes:
        s = (c.replace('Soybean_', 'Soy_')
              .replace('_Mosaic_Virus', '_MV')
              .replace('_necrosis_virus', '_nv')
              .replace('_Streak_Virus', '_SV')
              .replace('_leaf_spot', '_ls')
              .replace('_brown_spot', '_bs')
              .replace('_damping_off', '_do')
              .replace('_death_syndrome', '_ds')
              .replace('_Pod_Mottle_virus', '_PMv')
              .replace('_Stem_Rot', '_SR')
              .replace('_stem_disorder', '_sd')
              .replace('_Seed_Stain', '_SS')
              .replace('_Cyst_Nematode', '_CN')
              .replace('_Dwarf', '_D'))
        short.append(s)

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=90, fontsize=11)
    ax.set_yticklabels(short, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=15)
    ax.set_ylabel('Actual (Ground Truth)', fontsize=15)
    if not title:
        title = f'Confusion Matrix ({n} classes, {len(all_results)} images)'
    ax.set_title(title, fontsize=16)

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val > 0:
                color = 'white' if val >= 2 else 'black'
                ax.text(j, i, str(val), ha='center', va='center', fontsize=10, fontweight='bold', color=color)

    # Blue border on diagonal
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor='blue', linewidth=1.5))

    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--title', default='')
    args = parser.parse_args()
    plot_confusion_matrix(args.results_dir, args.output, args.title)
