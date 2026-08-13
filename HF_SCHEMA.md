# SAGE HuggingFace dataset — relational schema (fixes reviewer 1rWS)

**Problem flagged:** `load_dataset(...)` currently exposes `image / crop / disease`
but **not** the organ and symptom fields advertised as central contributions.

**Fix:** ship the dataset as four linked tables (or one flat table with all
fields joined), keyed on deterministic ids so every advertised field is loadable
and every symptom claim is traceable to its source. This mirrors the
provenance/validation table in the paper.

## Tables & keys

### `images.parquet`  (one row per image)
| column | type | notes |
|--------|------|-------|
| `image_id` | str (PK) | stable id = `<crop>/<disease>/<filename>` |
| `crop` | str | |
| `disease` | str (FK → diseases.disease_id via crop+disease) | canonical class |
| `split` | str | `reference` \| `test` |
| `organ` | str | VLM-annotated (see validation table) |
| `source_dataset` | str (FK → provenance) | origin corpus |
| `license` | str | inherited from source |

### `diseases.parquet`  (one row per canonical disease)
| column | type | notes |
|--------|------|-------|
| `disease_id` | str (PK) | `<crop>/<disease>` |
| `crop`, `disease` | str | |
| `pathogen` | str | + `pathogen_source_url`, `pathogen_quote` |
| `disease_type` | str | |
| `affected_parts` | list[str] | organ index used by the agent |
| `look_alikes` | list[str] | |
| `kb_confidence` | str | high/medium/low (self-reported) |
| `num_sources` | int | |

### `symptoms.parquet`  (one row per (disease, field) claim — the provenance layer)
| column | type | notes |
|--------|------|-------|
| `disease_id` | str (FK) | |
| `field` | str | `summary` \| `diagnostic_features` \| `affected_parts` \| `pathogen` \| … |
| `value` | str | extracted claim |
| `source_url` | str | authoritative page |
| `source_quote` | str | verbatim supporting quote |

### `provenance.parquet`  (image → origin)
| column | type | notes |
|--------|------|-------|
| `image_id` | str (FK) | |
| `source_dataset` | str | e.g. PlantVillage, Bugwood, CDDM … |
| `original_filename` | str | |
| `license` | str | |

## Loader contract
```python
ds = load_dataset("tirtho149/SAGE", "images")      # config per table
dis = load_dataset("tirtho149/SAGE", "diseases")
# join: images.disease_id == diseases.disease_id ; symptoms.disease_id likewise
```
A single flattened `all` config (images ⨝ diseases ⨝ symptoms) should also be
provided so casual users get organ + symptom columns without joining.

## Build & upload (needs the maintainer's HF token — not run here)
`scripts/build_hf_tables.py` (scaffold) emits the four parquets from
`disease_registry/outputs/*/final_registry.json` (diseases + symptoms) and the
image provenance sheet (images + provenance). Upload with:
```bash
huggingface-cli login            # maintainer credential
python scripts/build_hf_tables.py --out hf_release/
huggingface-cli upload tirtho149/SAGE hf_release/ --repo-type dataset
```
Auto-annotated fields (`organ`, image–class match) are labelled as such in the
dataset card per the provenance/validation table, so users don't mistake them
for expert-verified.
