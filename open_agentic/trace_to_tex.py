"""Convert JSON reasoning traces to formatted LaTeX files.

Reads trace JSONs from results/ and produces .tex files that can be
\\input{} from main.tex. Each trace shows the agent's step-by-step
reasoning for a single test image.

Usage:
    python -m CyberVisionAg.open_agentic.trace_to_tex
"""

import json
import re
import textwrap
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results" / "open_agentic"
TRACES_OUT = Path(__file__).resolve().parents[2] / "writing" / "69aae430e8bdcbd9056bf911" / "traces"


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters and strip control chars."""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = text.replace('**', '')
    for char, repl in [('&', r'\&'), ('%', r'\%'), ('$', r'\$'),
                       ('#', r'\#'), ('_', r'\_')]:
        text = text.replace(char, repl)
    return text


def _clean_text(text: str) -> str:
    """Clean agent text for display: remove JSON blocks, control chars, trim."""
    # Strip control characters (backspace, etc.)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Remove ```json ... ``` blocks
    text = re.sub(r'```json\s*\{.*?\}\s*```', '', text, flags=re.DOTALL)
    # Remove standalone JSON predictions
    text = re.sub(r'\{[^}]*"prediction"[^}]*\}', '', text)
    return text.strip()


def trace_to_tex(
    crop: str,
    kb_source: str,
    model: str,
    k: int,
    image_name: str,
    k_label=None,
) -> str:
    """Convert one trace to a LaTeX string."""
    # Load result
    kdir = k_label or f"k{k}"
    result_path = RESULTS_DIR / crop / kb_source / model / kdir / f"{image_name}.json"
    trace_path = RESULTS_DIR / crop / kb_source / model / kdir / "traces" / f"{image_name}.json"

    if not result_path.exists() or not trace_path.exists():
        return f"% Trace not found: {result_path}"

    result = json.loads(result_path.read_text())
    trace = json.loads(trace_path.read_text())

    prediction = result.get("prediction", "?")
    ground_truth = result.get("ground_truth", "?")
    correct = result.get("correct", False)
    confidence = result.get("confidence", 0)
    cost = result.get("cost_usd", 0)
    num_turns = result.get("num_turns", len(trace))

    outcome = r"\textcolor{green!60!black}{\textbf{Correct}}" if correct else r"\textcolor{red}{\textbf{Incorrect}}"

    # Copy test image (small thumbnail) to traces/images/
    img_tex = ""
    # image_name is ClassName__ImageStem, extract both parts
    if "__" in image_name:
        test_class, img_stem = image_name.split("__", 1)
    else:
        test_class = result.get("test_image", ground_truth)
        img_stem = image_name
    # Try prepared dataset first, then curated dataset
    PREPARED_DIR = SCRIPT_DIR.parent / "Prepared_Dataset"
    crop_short = crop.replace("_Diseases", "").replace("_Disease", "")
    search_paths = [
        PREPARED_DIR / f"{crop_short}_test" / test_class / f"{img_stem}.jpg",
        PREPARED_DIR / f"{crop_short}_test" / test_class / f"{img_stem}.jpeg",
        PREPARED_DIR / f"{crop_short}_test" / test_class / f"{img_stem}.png",
        SCRIPT_DIR.parent / "Curated_Local_Dataset" / "test" / crop / test_class / f"{img_stem}.jpg",
    ]
    src_img = None
    for p in search_paths:
        if p.exists():
            src_img = p
            break
    if src_img:
        from PIL import Image as PILImage
        img_out_dir = TRACES_OUT / "images"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        img_out = img_out_dir / f"{image_name}.jpg"
        im = PILImage.open(src_img)
        im.thumbnail((150, 150))
        im.save(img_out, quality=80)
        img_tex = rf"\includegraphics[height=2.5cm]{{traces/images/{image_name}.jpg}}"

    # Build step-by-step content
    steps = []
    step_num = 0
    for item in trace:
        typ = item.get("type", "")
        if typ == "tool_use":
            tool = item.get("tool", "")
            file_path = item.get("input", {}).get("file_path", "")
            fname = Path(file_path).stem if file_path else "?"
            # Classify what was read
            if "classify" in fname or "test" in fname.lower():
                action = "Observe test image"
            elif "confusion_guides" in file_path:
                action = f"Read calibration file: {fname}"
            else:
                action = f"View reference: {fname}"
            step_num += 1
            steps.append(f"\\item[Step {step_num}:] \\texttt{{{_escape_latex(action)}}}")
        elif typ == "text":
            content = _clean_text(item.get("content", ""))
            if not content:
                continue
            step_num += 1
            escaped = _escape_latex(content)
            steps.append(f"\\item[Step {step_num}:] {escaped}")

    steps_tex = "\n    ".join(steps)

    # Format config header
    crop_display = crop.replace("_", " ")
    kb_display = kb_source if kb_source != "none" else "no KB"
    image_display = image_name.replace("_", r"\_")

    tex = textwrap.dedent(rf"""
    \begin{{tcolorbox}}[
        breakable, colback=gray!3, colframe=gray!50, boxrule=0.5pt,
        title={{\small \textbf{{Reasoning Trace:}} {_escape_latex(crop_display)} — {image_display}}},
        fonttitle=\small,
    ]
    \small
    \begin{{minipage}}[t]{{0.65\linewidth}}
    \begin{{tabular}}{{@{{}}ll@{{}}}}
        \textbf{{Model:}} & {model} \\
        \textbf{{KB source:}} & {kb_display} \\
        \textbf{{Reference budget (k):}} & {k} \\
        \textbf{{Prediction:}} & {_escape_latex(prediction)} \\
        \textbf{{Ground truth:}} & {_escape_latex(ground_truth)} \\
        \textbf{{Outcome:}} & {outcome} \\
        \textbf{{Confidence:}} & {confidence:.2f} \\
    \end{{tabular}}
    \end{{minipage}}%
    \hfill
    \begin{{minipage}}[t]{{0.30\linewidth}}
    \centering
    \fbox{{{img_tex}}}\\[2pt]
    {{\scriptsize \textit{{Test image}}}}
    \end{{minipage}}

    \vspace{{4pt}}
    \hrule
    \vspace{{4pt}}

    \begin{{description}}[style=nextline, leftmargin=1.5cm, font=\small\bfseries]
    {steps_tex}
    \end{{description}}
    \end{{tcolorbox}}
    """).strip()

    return tex


def find_examples(crop, kb_source, model, k, n_correct=1, n_incorrect=1):
    """Find correct and incorrect examples from a config."""
    result_dir = RESULTS_DIR / crop / kb_source / model / f"k{k}"
    if not result_dir.exists():
        return [], []

    correct = []
    incorrect = []
    for f in sorted(result_dir.glob("*.json")):
        if f.name == "summary.json":
            continue
        d = json.loads(f.read_text())
        trace_f = result_dir / "traces" / f.name
        if not trace_f.exists():
            continue
        if d.get("correct") and len(correct) < n_correct:
            correct.append(f.stem)
        elif not d.get("correct") and len(incorrect) < n_incorrect:
            incorrect.append(f.stem)
        if len(correct) >= n_correct and len(incorrect) >= n_incorrect:
            break
    return correct, incorrect


def main():
    TRACES_OUT.mkdir(parents=True, exist_ok=True)

    # Preamble needed in main.tex: \usepackage{tcolorbox}
    # Check and remind
    print("TRACE TO TEX CONVERTER")
    print(f"  Output: {TRACES_OUT}")
    print()
    print("  NOTE: main.tex needs \\usepackage{tcolorbox} for trace boxes.")
    print()

    # ── In-text trace: one correct example, soybean, internet KB, k=4 ────
    intext_key = None
    correct, incorrect = find_examples("Soybean_Diseases", "internet", "sonnet", 4)
    if correct:
        intext_key = ("Soybean_Diseases", "internet", "sonnet", 4, correct[0])
        tex = trace_to_tex(*intext_key)
        out = TRACES_OUT / "trace_intext.tex"
        out.write_text(tex)
        print(f"  In-text trace (correct): {correct[0]} → {out.name}")

    # ── Appendix traces: ~10 across crops/configs ────────────────────────
    appendix_configs = [
        # (crop, kb_source, model, k)
        ("Soybean_Diseases", "internet", "sonnet", 4),
        ("Soybean_Diseases", "none", "sonnet", 4),
        ("Soybean_Diseases", "internet", "opus", 8),
        ("Corn_Diseases", "internet", "sonnet", 4),
        ("Corn_Diseases", "none", "sonnet", 8),
        ("Corn_Diseases", "internet", "opus", 8),
        ("Mango_Leaf_Disease", "internet", "sonnet", 4),
        ("Mango_Leaf_Disease", "none", "sonnet", 8),
    ]

    appendix_traces = []
    for crop, kb, model, k in appendix_configs:
        corr, incorr = find_examples(crop, kb, model, k, n_correct=1, n_incorrect=1)
        for name in corr:
            key = (crop, kb, model, k, name)
            if key != intext_key:  # skip the in-text trace
                appendix_traces.append(key)
        for name in incorr:
            appendix_traces.append((crop, kb, model, k, name))

    # Write individual trace files + a combined appendix file
    appendix_parts = []
    for i, (crop, kb, model, k, name) in enumerate(appendix_traces):
        tex = trace_to_tex(crop, kb, model, k, name)
        fname = f"trace_appendix_{i+1:02d}.tex"
        (TRACES_OUT / fname).write_text(tex)
        appendix_parts.append(f"\\input{{traces/{fname}}}")
        print(f"  Appendix trace {i+1}: {crop}/{kb}/{model}/k{k}/{name}")

    # Write combined appendix input file
    appendix_tex = "% Auto-generated appendix traces — do not edit manually\n"
    appendix_tex += "% Regenerate with: python -m CyberVisionAg.open_agentic.trace_to_tex\n\n"
    appendix_tex += "\n\\vspace{8pt}\n".join(appendix_parts)
    (TRACES_OUT / "traces_appendix.tex").write_text(appendix_tex)
    print(f"\n  Combined appendix: traces_appendix.tex ({len(appendix_traces)} traces)")


if __name__ == "__main__":
    main()
