"""Generate confusion matrix heatmap from eval results.

Follows plotting_instructions.md guidelines (Arial font, dpi=300, tight layout).
Designed at actual display size (~3.25in for half-textwidth in paper).
"""
import json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Abbreviations to keep labels short
ABBREVS = [
    ('Soybean_', 'Soy '),
    ('_Mosaic_Virus', ' MV'),
    ('_necrosis_virus', ' nv'),
    ('_Streak_Virus', ' SV'),
    ('_leaf_spot', ' ls'),
    ('_brown_spot', ' bs'),
    ('_damping_off', ' do'),
    ('_death_syndrome', ' ds'),
    ('_Pod_Mottle_virus', ' PMv'),
    ('_Stem_Rot', ' SR'),
    ('_stem_disorder', ' sd'),
    ('_Seed_Stain', ' SS'),
    ('_Cyst_Nematode', ' CN'),
    ('_Dwarf_', ' D '),
    ('_stalk_rot', ' SR'),
    ('_leaf_spot_top_dieback', ' ls/td'),
    ('_leaf_blight', ' LB'),
    ('_corn_leaf_blight', ' CLB'),
    ('_brown_spot', ' bs'),
    ('_stalk_rot', ' SR'),
]


def _shorten(name: str) -> str:
    """Shorten a class name for plot labels."""
    s = name
    for old, new in ABBREVS:
        s = s.replace(old, new)
    s = s.replace('_', ' ')
    return s


def plot_confusion_matrix(
    result_dir: str, output_path: str, title: str = "",
    highlight_classes: list[str] | None = None,
):
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

    short = [_shorten(c) for c in classes]

    # Design at actual display size
    if n <= 10:
        fig_size = (3.5, 3.2)
        tick_fs, label_fs, title_fs, cell_fs = 8, 9, 10, 10
    else:
        fig_size = (3.5, 3.2)
        tick_fs, label_fs, title_fs, cell_fs = 5, 7, 8, 5

    plt.rcParams.update({'font.family': 'Arial'})
    fig, ax = plt.subplots(figsize=fig_size)

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=90, fontsize=tick_fs)
    ax.set_yticklabels(short, fontsize=tick_fs)
    ax.set_xlabel('Predicted', fontsize=label_fs)
    ax.set_ylabel('Actual (Ground Truth)', fontsize=label_fs)
    if not title:
        title = f'Confusion Matrix ({n} classes)'
    ax.set_title(title, fontsize=title_fs)

    # Cell values
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val > 0:
                color = 'white' if val >= 2 else 'black'
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=cell_fs, fontweight='bold', color=color)

    # Diagonal border
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor='blue', linewidth=1))

    # Highlight specific columns (classes whose over-prediction was reduced)
    if highlight_classes:
        for cls in highlight_classes:
            if cls in class_to_idx:
                idx = class_to_idx[cls]
                # Highlight column (predicted)
                ax.add_patch(plt.Rectangle((idx - 0.5, -0.5), 1, n,
                                            fill=False, edgecolor='limegreen',
                                            linewidth=1.5, linestyle='-'))
                ax.get_xticklabels()[idx].set_fontweight('bold')
                ax.get_xticklabels()[idx].set_color('green')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=tick_fs)
    cbar.set_label('Count', fontsize=label_fs - 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def compare_confusion(base_dir: str, ag_dir: str) -> dict:
    """Print and return per-class accuracy changes and over-prediction shifts."""
    base_dir, ag_dir = Path(base_dir), Path(ag_dir)

    stats = {}
    for label, d in [('base', base_dir), ('ag', ag_dir)]:
        correct = defaultdict(int)
        total = defaultdict(int)
        wrong = defaultdict(lambda: defaultdict(int))
        for f in sorted(d.glob('*.json')):
            if f.name == 'summary.json': continue
            r = json.load(open(f))
            gt = r['ground_truth']
            total[gt] += 1
            if r.get('correct'): correct[gt] += 1
            elif r.get('prediction'): wrong[r['prediction']][gt] += 1
        stats[label] = (correct, total, wrong)

    bc, bt, bw = stats['base']
    ac, at, aw = stats['ag']

    # Per-class accuracy changes
    all_classes = sorted(set(list(bt.keys()) + list(at.keys())))
    improved = []
    worsened = []
    print("\nPer-class accuracy changes:")
    for cls in all_classes:
        ba = bc[cls]/bt[cls]*100 if bt[cls] else 0
        aa = ac[cls]/at[cls]*100 if at[cls] else 0
        delta = aa - ba
        if abs(delta) > 0.1:
            marker = "+++" if delta > 0 else "---"
            print(f"  {cls:35s} {bc[cls]}/{bt[cls]} -> {ac[cls]}/{at[cls]} ({delta:+.0f}pp) {marker}")
            if delta > 0: improved.append((cls, delta))
            else: worsened.append((cls, delta))

    # Over-prediction changes
    print("\nOver-prediction changes:")
    all_preds = sorted(set(list(bw.keys()) + list(aw.keys())))
    overpred_reduced = []
    for pred in all_preds:
        b = sum(bw[pred].values())
        a = sum(aw[pred].values())
        if a != b:
            print(f"  {pred:35s} {b}x -> {a}x ({a-b:+d})")
            if a < b:
                overpred_reduced.append((pred, b - a))

    # Suggest highlight: single column with biggest over-prediction reduction
    highlight = [cls for cls, _ in sorted(overpred_reduced, key=lambda x: -x[1])[:1]]

    print(f"\nSuggested column highlights (over-prediction reduced): {highlight}")
    return {"improved": improved, "worsened": worsened, "highlight": highlight}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--title', default='')
    parser.add_argument('--compare-with', default=None,
                        help='Compare with baseline dir (prints changes, suggests highlights)')
    parser.add_argument('--highlight', nargs='*', default=None,
                        help='Class names to highlight in the plot')
    args = parser.parse_args()

    highlights = args.highlight
    if args.compare_with and not highlights:
        info = compare_confusion(args.compare_with, args.results_dir)
        highlights = info['highlight']

    plot_confusion_matrix(args.results_dir, args.output, args.title,
                          highlight_classes=highlights)
