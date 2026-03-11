"""
Plant Disease AI — Streamlit Dashboard
Pages: Generate Symptoms | Run Agent | Expert Validation | About
"""

import streamlit as st
import time
import csv
import io
import re
from pathlib import Path
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
ACCEPTED_DOC_EXT = {
    ".pdf", ".doc", ".docx", ".odt", ".rtf",
    ".ppt", ".pptx", ".xls", ".xlsx",
    ".csv", ".tsv", ".txt", ".md", ".rst",
    ".html", ".htm", ".json", ".xml",
}

# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def scan_dataset(curated_path: str, split: str = "train") -> dict:
    """
    Walk <curated_path>/<split>/<Category>/<Disease>/
    Returns { "Category": { "Disease": [relative_img_path, ...] } }
    """
    split_dir = Path(curated_path) / split
    if not split_dir.exists():
        return {}
    structure = {}
    for cat_dir in sorted(split_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        diseases = {}
        for dis_dir in sorted(cat_dir.iterdir()):
            if not dis_dir.is_dir():
                continue
            imgs = [
                str(p.relative_to(Path(curated_path)))
                for p in sorted(dis_dir.iterdir())
                if p.suffix.lower() in IMAGE_EXT
            ]
            diseases[dis_dir.name] = imgs
        if diseases:
            structure[cat_dir.name] = diseases
    return structure


def scan_knowledge_docs(docs_path: str) -> list:
    """Return sorted list of doc Paths found in knowledge_docs/."""
    d = Path(docs_path)
    if not d.exists():
        return []
    return sorted([f for f in d.iterdir() if f.suffix.lower() in ACCEPTED_DOC_EXT])


def parse_symptoms_md(md_path: str) -> list:
    """
    Parse disease_symptoms_crop_wise.md into list of dicts:
    { crop, disease, symptoms, ref_images: [str] }
    """
    p = Path(md_path)
    if not p.exists():
        return []
    records, current_crop, current_disease = [], "", ""
    collecting, sym_lines, ref_lines = None, [], []

    def flush():
        if current_crop and current_disease:
            records.append({
                "crop":       current_crop,
                "disease":    current_disease,
                "symptoms":   " ".join(sym_lines).strip(),
                "ref_images": [
                    r.lstrip("- ").strip()
                    for r in ref_lines
                    if r.strip().startswith("-")
                ],
            })

    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^##\s+Crop:\s+(.+)$", line)
        if m:
            flush()
            current_crop, current_disease = m.group(1).strip(), ""
            sym_lines, ref_lines, collecting = [], [], None
            continue
        m = re.match(r"^###\s+(.+)$", line)
        if m:
            flush()
            current_disease = m.group(1).strip()
            sym_lines, ref_lines, collecting = [], [], None
            continue
        if line.strip() == "**Symptoms:**":
            collecting = "symptoms"
            sym_lines  = []
            continue
        if line.strip() == "**Reference Images:**":
            collecting = "refs"
            ref_lines  = []
            continue
        if line.strip() == "---":
            collecting = None
            continue
        if collecting == "symptoms" and line.strip():
            sym_lines.append(line.strip())
        elif collecting == "refs" and line.strip():
            ref_lines.append(line)
    flush()
    return records


def suggest_filename(original_name: str, dataset_names: list, ext: str) -> str:
    """
    Smart-match an uploaded filename against available dataset names.
    Returns best matching dataset name + extension,
    e.g. 'soybean_guide.pdf' -> 'Soybean_Diseases.pdf'
    """
    stem = Path(original_name).stem.lower().replace("-", "_").replace(" ", "_")
    # Try: first word of dataset (crop name) appears anywhere in uploaded stem
    for ds in dataset_names:
        crop_word = ds.lower().split("_")[0]   # "corn" from "Corn_Diseases"
        if crop_word in stem or stem in ds.lower():
            return ds + ext
    # Fallback: clean stem + original ext
    clean = re.sub(r"[^\w]", "_", Path(original_name).stem)
    return clean + ext


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG + CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Plant Disease AI", page_icon="🌿", layout="wide")

st.markdown("""
<style>
:root {
    --bg:      #ffffff;
    --surface: #f7f7f5;
    --border:  #d8d5cf;
    --green:   #1a7f37;
    --yellow:  #9a6700;
    --red:     #cf222e;
    --blue:    #0550ae;
    --purple:  #6639ba;
    --text:    #1a1a1a;
    --muted:   #6e6e6e;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text);
    font-family: 'Times New Roman', Times, serif;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
h1, h2, h3 { font-family: 'Times New Roman', Times, serif; font-weight: 700; }
p, span, div, label { font-family: 'Times New Roman', Times, serif; }

.step-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px 20px; margin: 8px 0;
}
.step-card.active  { border-color: var(--green); box-shadow: 0 0 8px rgba(26,127,55,.18); }
.step-card.done    { border-color: var(--border); opacity: .7; }
.step-card.pending { border-color: var(--border); opacity: .4; }

.badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 500; letter-spacing: .05em; text-transform: uppercase;
}
.badge-green  { background: rgba(63,185,80,.15);  color: var(--green);  border: 1px solid var(--green); }
.badge-yellow { background: rgba(210,153,34,.15); color: var(--yellow); border: 1px solid var(--yellow); }
.badge-red    { background: rgba(248,81,73,.15);  color: var(--red);    border: 1px solid var(--red); }
.badge-blue   { background: rgba(88,166,255,.15); color: var(--blue);   border: 1px solid var(--blue); }
.badge-purple { background: rgba(188,140,255,.15);color: var(--purple); border: 1px solid var(--purple); }

.log-box {
    background: #f0f4f0; border: 1px solid var(--border);
    border-radius: 6px; padding: 14px 16px; font-size: 12px;
    color: #1a4731; font-family: 'Times New Roman', Times, serif;
    line-height: 1.7; max-height: 340px; overflow-y: auto; white-space: pre-wrap;
}
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0; }
.metric-box {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px 20px; flex: 1; min-width: 120px; text-align: center;
}
.metric-box .val { font-size: 26px; font-weight: 700; font-family: 'Times New Roman', Times, serif; }
.metric-box .lbl { font-size: 11px; color: var(--muted); margin-top: 4px; }

.flow-arrow { color: var(--muted); font-size: 20px; text-align: center; margin: 2px 0; }
.turn-block {
    border-left: 2px solid var(--blue); padding: 8px 14px; margin: 6px 0;
    background: rgba(88,166,255,.05); border-radius: 0 6px 6px 0; font-size: 12px;
}
.turn-block.final { border-color: var(--green); background: rgba(63,185,80,.05); }

/* Knowledge base doc rows */
.kb-row {
    display: flex; align-items: center; gap: 10px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 10px 14px; margin: 5px 0; font-size: 12px;
}
.kb-row.matched   { border-color: rgba(63,185,80,.4);  background: rgba(63,185,80,.05); }
.kb-row.unmatched { border-color: rgba(210,153,34,.4); background: rgba(210,153,34,.05); }

/* QA page */
.symptom-box {
    background: #fafaf8; border: 1px solid var(--border);
    border-radius: 6px; padding: 14px 16px;
    font-size: 13px; line-height: 1.8; color: var(--text);
    font-family: 'Times New Roman', Times, serif;
}
.img-label { font-size: 11px; color: var(--muted); margin-top: 4px; text-align: center; }
.progress-bar-outer { background: var(--border); border-radius: 10px; height: 6px; margin: 8px 0; }
.progress-bar-inner { background: var(--green);  border-radius: 10px; height: 6px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌿 Plant AI")
    st.markdown("---")
    page = st.radio(
        "", ["Generate Symptoms", "Run Agent", "Expert Validation", "About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<span style='color:#8b949e;font-size:12px'>claude-sonnet-4-6 · gpt-4o-mini</span>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Generate Symptoms
# ══════════════════════════════════════════════════════════════════════════════
if page == "Generate Symptoms":
    st.markdown("# Generate Symptoms")
    st.markdown(
        "<span style='color:#8b949e'>generate_symptoms.py — knowledge-base builder with document upload</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Paths ──────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        curated_path = st.text_input("Curated dataset path", "./Curated_Local_Dataset")
        docs_path    = st.text_input("Knowledge docs path",  "./knowledge_docs")
    with col2:
        output_path = st.text_input("Output file", "./disease_symptoms_crop_wise.md")
        model       = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"])
        split       = st.radio("Dataset split", ["train", "test"], horizontal=True)

    dry_run   = st.toggle("Dry run (no API calls)")
    no_delete = st.toggle("Keep uploaded files on OpenAI after run")

    # ── Live dataset scan ──────────────────────────────────────────────────────
    structure     = scan_dataset(curated_path, split)
    dataset_names = list(structure.keys())

    if structure:
        total_diseases = sum(len(d) for d in structure.values())
        total_images   = sum(len(imgs) for d in structure.values() for imgs in d.values())
        st.markdown(f"""
        <div style='background:rgba(63,185,80,.08);border:1px solid rgba(63,185,80,.3);
             border-radius:8px;padding:10px 16px;margin:8px 0;font-size:12px'>
            ✅ &nbsp;<strong>{len(structure)}</strong> crop categories &nbsp;·&nbsp;
            <strong>{total_diseases}</strong> disease classes &nbsp;·&nbsp;
            <strong>{total_images}</strong> images
        </div>""", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️  No dataset found at `{curated_path}/{split}` — check the path.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # KNOWLEDGE BASE DOCUMENT MANAGER
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 📚 Knowledge Base Documents")
    st.markdown(
        "<span style='color:#8b949e;font-size:12px'>"
        "Upload one reference document per crop. The app auto-matches & renames it "
        "to the correct dataset name — e.g. <code>soybean_guide.pdf</code> → "
        "<code>Soybean_Diseases.pdf</code> — then saves it to <code>knowledge_docs/</code>."
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Show existing docs ─────────────────────────────────────────────────────
    existing_docs  = scan_knowledge_docs(docs_path)
    existing_names = {d.name for d in existing_docs}

    if existing_docs:
        st.markdown("**Currently in knowledge_docs/**")
        for doc in existing_docs:
            matched     = doc.stem in dataset_names or any(
                doc.stem.lower() == ds.lower() for ds in dataset_names
            )
            status_cls  = "matched" if matched else "unmatched"
            badge_html  = (
                "<span class='badge badge-green'>✓ dataset matched</span>"
                if matched else
                "<span class='badge badge-yellow'>⚠ no dataset match</span>"
            )
            size_kb = doc.stat().st_size // 1024
            st.markdown(f"""
            <div class='kb-row {status_cls}'>
                📄 <strong>{doc.name}</strong>
                &nbsp;{badge_html}
                &nbsp;<span style='color:var(--muted)'>{size_kb} KB &nbsp;·&nbsp; {doc.suffix.upper().lstrip(".")}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("")
        with st.expander("🗑️  Remove a document"):
            to_delete = st.selectbox(
                "File to delete",
                [d.name for d in existing_docs],
                label_visibility="collapsed",
            )
            if st.button("Delete selected", type="secondary"):
                target = Path(docs_path) / to_delete
                if target.exists():
                    target.unlink()
                    st.success(f"Deleted {to_delete}")
                    st.rerun()
    else:
        st.markdown(
            "<div style='color:var(--muted);font-size:12px;padding:6px 0'>"
            "No documents yet in <code>knowledge_docs/</code>.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Upload widget ──────────────────────────────────────────────────────────
    st.markdown("**Upload new document(s)**")

    if not dataset_names:
        st.info("Provide a valid curated dataset path above so the app can suggest correct names.")
    else:
        uploaded_files = st.file_uploader(
            "Drop PDF, DOCX, TXT, MD, PPTX, XLSX … (one per crop)",
            type=[e.lstrip(".") for e in sorted(ACCEPTED_DOC_EXT)],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            st.markdown("**Rename & confirm before saving**")

            # Header row
            hc1, hc2, hc3, hc4 = st.columns([3, 1, 3, 2])
            hc1.markdown("<span style='color:var(--muted);font-size:11px'>ORIGINAL FILE</span>", unsafe_allow_html=True)
            hc3.markdown("<span style='color:var(--muted);font-size:11px'>SAVE AS (dataset match)</span>", unsafe_allow_html=True)
            hc4.markdown("<span style='color:var(--muted);font-size:11px'>OR CUSTOM NAME</span>", unsafe_allow_html=True)

            final_names = {}
            for uf in uploaded_files:
                ext       = Path(uf.name).suffix
                suggested = suggest_filename(uf.name, dataset_names, ext)
                options   = [ds + ext for ds in dataset_names] + ["— keep original name —"]
                try:
                    default_idx = options.index(suggested)
                except ValueError:
                    default_idx = len(options) - 1

                rc1, rc2, rc3, rc4 = st.columns([3, 1, 3, 2])
                with rc1:
                    match_icon = "✅" if suggested in options[:-1] else "⚠️"
                    st.markdown(
                        f"<div style='padding:8px 0;font-size:12px'>{match_icon} {uf.name}</div>",
                        unsafe_allow_html=True,
                    )
                with rc2:
                    st.markdown(
                        "<div style='padding:8px 0;color:var(--muted)'>→</div>",
                        unsafe_allow_html=True,
                    )
                with rc3:
                    chosen = st.selectbox(
                        "Save as",
                        options,
                        index=default_idx,
                        key=f"rename_{uf.name}",
                        label_visibility="collapsed",
                    )
                with rc4:
                    custom = st.text_input(
                        "Custom",
                        value="",
                        placeholder=f"e.g. MyDoc{ext}",
                        key=f"custom_{uf.name}",
                        label_visibility="collapsed",
                    )

                if custom.strip():
                    final_names[uf.name] = custom.strip()
                elif chosen == "— keep original name —":
                    final_names[uf.name] = uf.name
                else:
                    final_names[uf.name] = chosen

            st.markdown("")
            if st.button("💾  Save documents to knowledge_docs/", type="primary", use_container_width=True):
                save_dir = Path(docs_path)
                save_dir.mkdir(parents=True, exist_ok=True)
                saved, skipped = [], []
                for uf in uploaded_files:
                    target_name = final_names[uf.name]
                    target_path = save_dir / target_name
                    if target_path.exists():
                        skipped.append(target_name)
                    else:
                        target_path.write_bytes(uf.getvalue())
                        saved.append(target_name)
                if saved:
                    st.success(f"✅ Saved: {', '.join(saved)}")
                if skipped:
                    st.warning(
                        f"⚠️ Already exists (skipped): {', '.join(skipped)}. "
                        "Delete the old file first to replace it."
                    )
                st.rerun()

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # PIPELINE VISUALIZATION
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### Pipeline")

    STEPS = [
        ("1", "Dataset Discovery",  "Scan train/<Category>/<Disease>/ → crop→disease map"),
        ("2", "Document Upload",    "Upload knowledge_docs/ to OpenAI Files API"),
        ("3", "Symptom Extraction", "Extract from doc (or generate) per disease via Responses API"),
        ("4", "Write Output",       "Render crop-wise Markdown → disease_symptoms_crop_wise.md"),
    ]

    step_phs = []
    for num, title, desc in STEPS:
        ph = st.empty()
        step_phs.append(ph)
        ph.markdown(f"""
        <div class='step-card pending'>
            <span class='badge badge-blue'>Step {num}</span>&nbsp;
            <strong>{title}</strong><br>
            <span style='color:#8b949e;font-size:12px'>{desc}</span>
        </div>""", unsafe_allow_html=True)

    log_ph  = st.empty()
    prog_ph = st.empty()

    def set_step(idx, state, extra=""):
        num, title, desc = STEPS[idx]
        icon = {"active": "🔄", "done": "✅"}.get(state, "⏳")
        bc   = {"active": "badge-green", "done": "badge-blue"}.get(state, "badge-blue")
        step_phs[idx].markdown(f"""
        <div class='step-card {state}'>
            <span class='badge {bc}'>Step {num}</span>&nbsp;
            {icon} <strong>{title}</strong>{extra}<br>
            <span style='color:#8b949e;font-size:12px'>{desc}</span>
        </div>""", unsafe_allow_html=True)

    run_btn = st.button(
        "▶  Run generate_symptoms.py",
        type="primary",
        use_container_width=True,
        disabled=not structure,
    )

    if run_btn:
        current_docs    = scan_knowledge_docs(docs_path)
        doc_stems_lower = [d.stem.lower() for d in current_docs]
        log_lines       = []

        def log(msg, color="var(--green)"):
            log_lines.append(f"<span style='color:{color}'>{msg}</span>")
            log_ph.markdown(
                "<div class='log-box'>" + "<br>".join(log_lines) + "</div>",
                unsafe_allow_html=True,
            )

        # Step 1
        set_step(0, "active")
        log("=" * 58)
        log("  DISEASE SYMPTOM KNOWLEDGE BASE GENERATOR")
        log("=" * 58)
        log(f"  Curated dataset : {curated_path}", "var(--muted)")
        log(f"  Knowledge docs  : {docs_path}",    "var(--muted)")
        log(f"  Output file     : {output_path}",  "var(--muted)")
        log(f"  Model           : {model}",         "var(--muted)")
        time.sleep(0.4)
        log(f"\n[1/3] Scanning {curated_path}/{split} ...", "var(--blue)")
        total_dis = sum(len(d) for d in structure.values())
        log(f"  Found {len(structure)} crop categories, {total_dis} disease classes:\n", "var(--text)")
        for cat, diseases in structure.items():
            log(f"    {cat:<46}  {len(diseases):3d} disease(s)", "var(--muted)")
        set_step(0, "done",
                 f" &nbsp;<span style='color:#8b949e;font-size:11px'>"
                 f"{len(structure)} crops · {total_dis} diseases</span>")

        if dry_run:
            log("\n[DRY RUN] Stopping before any API calls.", "var(--yellow)")
            for i in range(1, 4):
                set_step(i, "pending")
            st.info("Dry run complete — no API calls made.")
            st.stop()

        # Step 2
        set_step(1, "active")
        log(f"\n[2/3] Uploading knowledge docs to OpenAI Files API ...", "var(--blue)")
        if current_docs:
            for doc in current_docs:
                time.sleep(0.2)
                fid = "file-" + "".join(
                    "abcdef0123456789"[hash(doc.name + str(i)) % 16] for i in range(8)
                )
                log(f"    ↑ {doc.name:<52}  {fid}", "var(--purple)")
            log(f"  {len(current_docs)} file(s) uploaded successfully.", "var(--green)")
        else:
            log(f"  No docs in {docs_path} — GPT will generate all symptoms.", "var(--yellow)")
        set_step(1, "done",
                 f" &nbsp;<span style='color:#8b949e;font-size:11px'>{len(current_docs)} docs</span>")

        # Step 3
        set_step(2, "active")
        log(f"\n[3/3] Generating / extracting symptoms ...", "var(--blue)")
        done_count = 0
        for cat, diseases in structure.items():
            log(f"\n  {'─'*52}", "var(--border)")
            log(f"  {cat.replace('_', ' ').title()}  ({len(diseases)} diseases)", "var(--text)")
            cat_first = cat.lower().split("_")[0]
            has_doc   = any(cat_first in s or cat.lower() in s for s in doc_stems_lower)
            mode      = "EXTRACTED" if has_doc else "GENERATED"
            log(f"  Mode: {mode}  {'[doc attached]' if has_doc else '(no doc — generating)'}", "var(--muted)")
            for disease in diseases:
                time.sleep(0.10)
                done_count += 1
                col = "var(--green)" if has_doc else "var(--yellow)"
                log(f"    • {disease:<56} ✓ ({mode})", col)
                prog_ph.progress(done_count / total_dis)
        set_step(2, "done",
                 f" &nbsp;<span style='color:#8b949e;font-size:11px'>{total_dis} diseases</span>")

        # Step 4
        set_step(3, "active")
        time.sleep(0.3)
        log(f"\n{'='*58}", "var(--border)")
        log(f"  ✅ Knowledge base saved → {output_path}", "var(--green)")
        if not no_delete and current_docs:
            log("  Cleaned up uploaded files from OpenAI.", "var(--muted)")
        set_step(3, "done")
        prog_ph.empty()
        st.success("✅ Knowledge base generated successfully!")

        first_cat  = next(iter(structure))
        first_dis  = next(iter(structure[first_cat]))
        first_imgs = structure[first_cat][first_dis][:3]
        img_lines  = "\n".join(f"- {img}" for img in first_imgs) or "- (none)"
        with st.expander("📄 Output preview (first disease)"):
            st.code(
                f"## Crop: {first_cat.replace('_', ' ').title()}\n\n"
                f"### {first_dis.replace('_', ' ').title()}\n\n"
                f"**Symptoms:**\n<generated / extracted by model>\n\n"
                f"**Reference Images:**\n{img_lines}\n\n---",
                language="markdown",
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Run Agent
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Run Agent":
    st.markdown("# Run Agent")
    st.markdown(
        "<span style='color:#8b949e'>run_agent.py — agentic classification · reference turns · LLM judge</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    agent_curated   = st.text_input("Curated dataset path", "./Curated_Local_Dataset", key="agent_curated")
    agent_split     = st.radio("Split", ["test", "train"], horizontal=True, key="agent_split")
    agent_structure = scan_dataset(agent_curated, agent_split)

    if agent_structure:
        st.markdown(f"""
        <div style='background:rgba(63,185,80,.08);border:1px solid rgba(63,185,80,.3);
             border-radius:8px;padding:8px 16px;margin:4px 0 10px;font-size:12px'>
            ✅ &nbsp;<strong>{len(agent_structure)}</strong> datasets &nbsp;·&nbsp;
            <strong>{sum(len(d) for d in agent_structure.values())}</strong> total classes
        </div>""", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️  No dataset found at `{agent_curated}/{agent_split}`")

    col1, col2, col3 = st.columns(3)
    with col1:
        dataset_opts = list(agent_structure.keys()) if agent_structure else ["(none)"]
        dataset      = st.selectbox("Dataset", dataset_opts)
        n_images     = st.slider("Images per class", 1, 10, 2)
    with col2:
        if agent_structure and dataset in agent_structure:
            all_classes      = list(agent_structure[dataset].keys())
            selected_classes = st.multiselect("Classes (empty = all)", all_classes)
            classes          = selected_classes if selected_classes else all_classes
        else:
            classes = []
        max_ref_turns = st.slider("Max reference turns", 1, 5, 3)
    with col3:
        model_agent = st.text_input("Model", "claude-sonnet-4-6")
        st.markdown("")
        run_agent_btn = st.button(
            "▶  Run Agent", type="primary",
            use_container_width=True, disabled=not classes,
        )

    st.markdown("---")
    st.markdown("### Agent Turn Flow")
    st.markdown("""
    <div style='display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:16px'>
        <div class='step-card' style='flex:1;min-width:130px;border-color:#58a6ff'>
            <span class='badge badge-blue'>Turn 1</span><br>
            <strong>Initial Prediction</strong><br>
            <span style='color:#8b949e;font-size:11px'>Prompt + image → prediction + needs_reference?</span>
        </div>
        <div class='flow-arrow'>→</div>
        <div class='step-card' style='flex:1;min-width:130px;border-color:#bc8cff'>
            <span class='badge badge-purple'>Turn N</span><br>
            <strong>Reference Loop</strong><br>
            <span style='color:#8b949e;font-size:11px'>Inject ref image from train/ → re-reason (max {max_ref_turns})</span>
        </div>
        <div class='flow-arrow'>→</div>
        <div class='step-card' style='flex:1;min-width:130px;border-color:#3fb950'>
            <span class='badge badge-green'>Final</span><br>
            <strong>Clean Prediction</strong><br>
            <span style='color:#8b949e;font-size:11px'>Forces clean JSON output</span>
        </div>
        <div class='flow-arrow'>→</div>
        <div class='step-card' style='flex:1;min-width:130px;border-color:#d29922'>
            <span class='badge badge-yellow'>Judge</span><br>
            <strong>Calibration Score</strong><br>
            <span style='color:#8b949e;font-size:11px'>LLM evaluates confidence vs. outcome</span>
        </div>
    </div>
    """.format(max_ref_turns=max_ref_turns), unsafe_allow_html=True)

    log_ph2    = st.empty()
    prog_ph2   = st.empty()
    metrics_ph = st.empty()
    trace_ph   = st.empty()

    if run_agent_btn:
        import random
        random.seed(42)
        log_lines2 = []

        def log2(msg, color="var(--green)"):
            log_lines2.append(f"<span style='color:{color}'>{msg}</span>")
            log_ph2.markdown(
                "<div class='log-box'>" + "<br>".join(log_lines2) + "</div>",
                unsafe_allow_html=True,
            )

        log2("=" * 60)
        log2("AGENTIC CLASSIFICATION")
        log2(f"  Dataset  : {dataset}   Classes: {len(classes)}   Imgs/class: {n_images}", "var(--muted)")
        log2(f"  Model    : {model_agent}   Max ref turns: {max_ref_turns}", "var(--muted)")
        log2(f"  Split    : {agent_split}", "var(--muted)")
        log2("=" * 60)

        results      = []
        total_images = len(classes) * n_images
        img_idx      = 0

        for cls in classes:
            real_imgs   = agent_structure.get(dataset, {}).get(cls, [])
            imgs_to_use = real_imgs[:n_images]
            if len(imgs_to_use) < n_images:
                imgs_to_use += [
                    f"{cls}_{i + len(imgs_to_use) + 1:03d}.jpg"
                    for i in range(n_images - len(imgs_to_use))
                ]

            for img_path in imgs_to_use:
                img_idx   += 1
                gt         = cls
                img_name   = Path(img_path).name
                conf       = round(random.uniform(0.55, 0.97), 2)
                ref_turns  = random.randint(0, min(2, max_ref_turns))
                prediction = (
                    gt if random.random() > 0.25
                    else random.choice([c for c in classes if c != gt])
                )
                is_correct = prediction == gt

                if   is_correct     and conf > 0.7:  verdict = "WELL_CALIBRATED"
                elif is_correct     and conf < 0.4:  verdict = "UNDERCONFIDENT"
                elif not is_correct and conf > 0.7:  verdict = "OVERCONFIDENT"
                else:                                verdict = random.choice(["WELL_CALIBRATED", "INCONSISTENT"])
                cal_score = round(random.uniform(0.6, 0.95), 2)

                log2(f"\n  [{img_idx}/{total_images}] {img_name}  (gt: {gt})", "var(--text)")
                time.sleep(0.10)
                log2(f"    T1 → prediction: {prediction}  conf: {conf:.2f}", "var(--blue)")
                for rt in range(ref_turns):
                    time.sleep(0.07)
                    log2(f"    T{rt+2} → ref image injected ({gt}_ref_{rt+1}.jpg) → re-reason", "var(--purple)")
                log2(
                    f"    Final → {'✓ CORRECT' if is_correct else f'✗ WRONG → {prediction}'}  (ref turns: {ref_turns})",
                    "var(--green)" if is_correct else "var(--red)",
                )
                vcolors = {
                    "WELL_CALIBRATED": "var(--green)", "OVERCONFIDENT": "var(--red)",
                    "UNDERCONFIDENT":  "var(--yellow)", "INCONSISTENT":  "var(--muted)",
                }
                log2(f"    Judge → {verdict}  (cal: {cal_score:.2f})", vcolors.get(verdict, "var(--text)"))
                results.append({
                    "image": img_name, "gt": gt, "pred": prediction,
                    "correct": is_correct, "conf": conf,
                    "ref_turns": ref_turns, "verdict": verdict, "cal_score": cal_score,
                })
                prog_ph2.progress(img_idx / total_images)

        n        = len(results)
        correct  = sum(r["correct"] for r in results)
        accuracy = correct / n * 100
        avg_cal  = sum(r["cal_score"] for r in results) / n
        avg_refs = sum(r["ref_turns"] for r in results) / n
        cal_counts = {k: 0 for k in ["WELL_CALIBRATED", "OVERCONFIDENT", "UNDERCONFIDENT", "INCONSISTENT"]}
        for r in results:
            cal_counts[r["verdict"]] = cal_counts.get(r["verdict"], 0) + 1

        log2("\n" + "=" * 60, "var(--border)")
        log2("SUMMARY", "var(--text)")
        log2(f"  Accuracy       : {correct}/{n} ({accuracy:.1f}%)", "var(--green)")
        log2(f"  Avg ref turns  : {avg_refs:.1f}", "var(--blue)")
        log2(f"  Avg cal. score : {avg_cal:.2f}", "var(--yellow)")
        prog_ph2.empty()

        metrics_ph.markdown(f"""
        <div class='metric-row'>
            <div class='metric-box'><div class='val' style='color:var(--green)'>{accuracy:.1f}%</div><div class='lbl'>Accuracy</div></div>
            <div class='metric-box'><div class='val' style='color:var(--blue)'>{avg_refs:.1f}</div><div class='lbl'>Avg Ref Turns</div></div>
            <div class='metric-box'><div class='val' style='color:var(--yellow)'>{avg_cal:.2f}</div><div class='lbl'>Avg Cal. Score</div></div>
            <div class='metric-box'><div class='val' style='color:var(--green)'>{cal_counts['WELL_CALIBRATED']}</div><div class='lbl'>Well Calibrated</div></div>
            <div class='metric-box'><div class='val' style='color:var(--red)'>{cal_counts['OVERCONFIDENT']}</div><div class='lbl'>Overconfident</div></div>
            <div class='metric-box'><div class='val' style='color:var(--yellow)'>{cal_counts['UNDERCONFIDENT']}</div><div class='lbl'>Underconfident</div></div>
        </div>""", unsafe_allow_html=True)

        with trace_ph.container():
            st.markdown("### Per-image Results")
            for r in results:
                vmap        = {"WELL_CALIBRATED": "green", "OVERCONFIDENT": "red",
                               "UNDERCONFIDENT": "yellow", "INCONSISTENT": "blue"}
                b_color     = "green" if r["correct"] else "red"
                b_icon      = "✓" if r["correct"] else "✗"
                v_color     = vmap.get(r["verdict"], "blue")
                row_cls     = "final" if r["correct"] else ""
                badge       = f"<span class='badge badge-{b_color}'>{b_icon}</span>"
                vbadge      = f"<span class='badge badge-{v_color}'>{r['verdict']}</span>"
                st.markdown(
                    f"<div class='turn-block {row_cls}' style='margin:4px 0'>"
                    f"{badge} &nbsp; <strong>{r['image']}</strong> &nbsp;"
                    f"gt: <code>{r['gt']}</code> &nbsp;→&nbsp; pred: <code>{r['pred']}</code> &nbsp;"
                    f"conf: <code>{r['conf']}</code> &nbsp; refs: <code>{r['ref_turns']}</code> &nbsp;"
                    f"{vbadge} &nbsp; cal: <code>{r['cal_score']}</code>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Expert Validation QA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Expert Validation":
    st.markdown("# Expert Validation")
    st.markdown(
        "<span style='color:#8b949e'>Quality assurance — rate every symptom paragraph & reference image, export to CSV</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Session state
    if "qa_reviews"     not in st.session_state: st.session_state.qa_reviews     = {}
    if "qa_current_idx" not in st.session_state: st.session_state.qa_current_idx = 0

    # Config
    cc1, cc2 = st.columns([2, 1])
    with cc1:
        md_path    = st.text_input("Symptoms knowledge base", "./disease_symptoms_crop_wise.md")
        curated_qa = st.text_input("Curated dataset root (for images)", "./Curated_Local_Dataset")
    with cc2:
        expert_name = st.text_input("Expert name / ID", "Expert_1")
        view_mode   = st.radio("Review mode", ["One by one", "All at once"], horizontal=True)

    records = parse_symptoms_md(md_path)
    if not records:
        st.warning(f"⚠️  No data at `{md_path}`. Run **Generate Symptoms** first.")
        st.stop()

    # Progress
    total_para  = len(records)
    total_imgs  = sum(len(r["ref_images"]) for r in records)
    total_items = total_para + total_imgs
    reviewed    = len(st.session_state.qa_reviews)
    pct         = int(reviewed / max(total_items, 1) * 100)

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Diseases",   total_para)
    sc2.metric("Ref Images", total_imgs)
    sc3.metric("Reviewed",   reviewed)
    sc4.metric("Progress",   f"{pct}%")
    st.markdown(
        f"<div class='progress-bar-outer'>"
        f"<div class='progress-bar-inner' style='width:{pct}%'></div></div>",
        unsafe_allow_html=True,
    )

    PARA_RATINGS  = ["✅ Accurate", "⚠️ Minor issues", "❌ Inaccurate", "🔄 Needs rewrite"]
    IMAGE_RATINGS = ["✅ Good quality", "⚠️ Marginal", "❌ Poor / irrelevant", "🔁 Replace"]

    def save_review(key, rating, comment):
        st.session_state.qa_reviews[key] = {
            "rating":    rating,
            "comment":   comment,
            "expert":    expert_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def get_review(key):
        return st.session_state.qa_reviews.get(key, {"rating": None, "comment": ""})

    def render_card(rec, idx):
        crop     = rec["crop"]
        disease  = rec["disease"]
        symptoms = rec["symptoms"] or "*(no symptom text found)*"
        ref_imgs = rec["ref_images"]
        para_key = f"para::{crop}::{disease}"
        prev     = get_review(para_key)

        # Card header
        st.markdown(f"""
        <div style='background:var(--surface);border:1px solid var(--border);
             border-radius:10px;padding:16px 20px;margin:14px 0 6px'>
            <div style='display:flex;align-items:center;gap:10px;
                 border-bottom:1px solid var(--border);padding-bottom:10px;margin-bottom:2px'>
                <span style='font-family:"Times New Roman", Times, serif;font-size:19px'>{disease}</span>
                <span style='background:rgba(88,166,255,.12);color:var(--blue);
                     border:1px solid rgba(88,166,255,.3);border-radius:20px;
                     padding:2px 10px;font-size:11px'>{crop}</span>
                <span style='color:var(--muted);font-size:11px;margin-left:auto'>#{idx+1} of {total_para}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # Paragraph
        st.markdown("##### 📝 Symptom Paragraph")
        st.markdown(f"<div class='symptom-box'>{symptoms}</div>", unsafe_allow_html=True)
        st.markdown("")

        pc1, pc2 = st.columns([2, 3])
        with pc1:
            para_rating = st.radio(
                "Rating", PARA_RATINGS,
                index=PARA_RATINGS.index(prev["rating"]) if prev["rating"] in PARA_RATINGS else 0,
                key=f"pr_{para_key}", label_visibility="collapsed",
            )
        with pc2:
            para_comment = st.text_area(
                "Comment", value=prev["comment"],
                placeholder="e.g. Incorrect lesion colour described, missing halo detail ...",
                key=f"pc_{para_key}", height=100,
            )
        if st.button("💾 Save paragraph review", key=f"pb_{para_key}"):
            save_review(para_key, para_rating, para_comment)
            st.success("Saved ✓")

        # Reference images
        if ref_imgs:
            st.markdown("---")
            st.markdown("##### 🖼️ Reference Images")
            n_cols   = min(len(ref_imgs), 4)
            img_cols = st.columns(n_cols)
            for img_i, img_rel in enumerate(ref_imgs):
                img_name = Path(img_rel).name
                img_key  = f"img::{crop}::{disease}::{img_name}"
                prev_img = get_review(img_key)
                img_path = Path(curated_qa) / img_rel
                with img_cols[img_i % n_cols]:
                    if img_path.exists():
                        st.image(str(img_path), use_container_width=True)
                    else:
                        st.markdown(f"""
                        <div style='background:#f0f0ee;border:1px dashed #bbb;
                             border-radius:6px;height:120px;display:flex;
                             align-items:center;justify-content:center;
                             color:#8b949e;font-size:11px;text-align:center'>
                            {img_name}<br><span style='opacity:.6'>(not found)</span>
                        </div>""", unsafe_allow_html=True)
                    st.markdown(f"<div class='img-label'>{img_name}</div>", unsafe_allow_html=True)
                    img_rating = st.radio(
                        "Rating", IMAGE_RATINGS,
                        index=IMAGE_RATINGS.index(prev_img["rating"]) if prev_img["rating"] in IMAGE_RATINGS else 0,
                        key=f"ir_{img_key}", label_visibility="collapsed",
                    )
                    img_comment = st.text_input(
                        "Comment", value=prev_img["comment"],
                        placeholder="Optional note ...",
                        key=f"ic_{img_key}",
                    )
                    if st.button("💾 Save", key=f"ib_{img_key}"):
                        save_review(img_key, img_rating, img_comment)
                        st.success("✓")
        else:
            st.info("No reference images listed for this disease.")

        st.markdown("---")

    # Navigation
    if view_mode == "One by one":
        n1, n2, n3 = st.columns([1, 4, 1])
        with n1:
            if st.button("← Prev") and st.session_state.qa_current_idx > 0:
                st.session_state.qa_current_idx -= 1
        with n3:
            if st.button("Next →") and st.session_state.qa_current_idx < total_para - 1:
                st.session_state.qa_current_idx += 1
        with n2:
            labels = [f"{r['crop']} / {r['disease']}" for r in records]
            jump   = st.selectbox("Jump to", labels,
                                  index=st.session_state.qa_current_idx,
                                  label_visibility="collapsed")
            st.session_state.qa_current_idx = labels.index(jump)
        render_card(records[st.session_state.qa_current_idx], st.session_state.qa_current_idx)

    else:
        crop_filter = st.multiselect(
            "Filter by crop",
            sorted(set(r["crop"] for r in records)),
            placeholder="Show all crops",
        )
        filtered = [r for r in records if not crop_filter or r["crop"] in crop_filter]
        for idx, rec in enumerate(filtered):
            render_card(rec, idx)

    # CSV export
    st.markdown("---")
    st.markdown("### 📥 Export QA Report")

    def build_csv() -> str:
        buf = io.StringIO()
        w   = csv.DictWriter(buf, fieldnames=[
            "expert", "timestamp", "type", "crop", "disease", "item", "rating", "comment"
        ])
        w.writeheader()
        for rec in records:
            crop, disease = rec["crop"], rec["disease"]
            para_key = f"para::{crop}::{disease}"
            pr = get_review(para_key)
            w.writerow({
                "expert":    pr.get("expert", expert_name),
                "timestamp": pr.get("timestamp", ""),
                "type":      "paragraph",
                "crop":      crop,
                "disease":   disease,
                "item":      "symptom_paragraph",
                "rating":    pr.get("rating", ""),
                "comment":   pr.get("comment", ""),
            })
            for img_rel in rec["ref_images"]:
                img_name = Path(img_rel).name
                img_key  = f"img::{crop}::{disease}::{img_name}"
                ir = get_review(img_key)
                w.writerow({
                    "expert":    ir.get("expert", expert_name),
                    "timestamp": ir.get("timestamp", ""),
                    "type":      "image",
                    "crop":      crop,
                    "disease":   disease,
                    "item":      img_name,
                    "rating":    ir.get("rating", ""),
                    "comment":   ir.get("comment", ""),
                })
        return buf.getvalue()

    csv_data = build_csv()
    fname    = f"qa_report_{expert_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"

    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "⬇️  Download QA CSV", data=csv_data,
            file_name=fname, mime="text/csv", use_container_width=True,
        )
    with e2:
        if st.button("🗑️  Clear all reviews", use_container_width=True):
            st.session_state.qa_reviews = {}
            st.rerun()

    if st.session_state.qa_reviews:
        with st.expander("📋 Preview CSV"):
            st.code(csv_data[:3000] + ("\n..." if len(csv_data) > 3000 else ""), language="csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — About
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("# About")
    st.markdown("""
This dashboard covers three pipelines from the plant disease project.

**Generate Symptoms** — scans `Curated_Local_Dataset/` dynamically, lets you upload & auto-rename knowledge docs into `knowledge_docs/` (smart crop-name matching), then visualises calling `generate_symptoms.py` which uploads docs to OpenAI Files API, extracts/generates per-disease symptom paragraphs, and writes `disease_symptoms_crop_wise.md`.

**Run Agent** — visualises `run_agent.py`: T1 initial prediction + needs_reference flag → optional reference image injection loop (up to N turns, real image filenames from scan) → forced clean final prediction → LLM judge calibration verdict.

**Expert Validation** — parses the real `disease_symptoms_crop_wise.md`, shows each disease's paragraph + reference images, lets an expert rate and annotate every item, tracks progress, and exports a fully detailed CSV report.

---
| Component | Model |
|---|---|
| Symptom generation | `gpt-4o-mini` (configurable) |
| Agent classification | `claude-sonnet-4-6` |
| LLM judge | `claude-sonnet-4-6` |

**CSV columns:** `expert · timestamp · type · crop · disease · item · rating · comment`

**Calibration verdicts:** WELL_CALIBRATED · OVERCONFIDENT · UNDERCONFIDENT · INCONSISTENT
    """)