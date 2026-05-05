# capsuled_data_prep

Self-contained capsule for running `prepare_dataset` on a remote cluster
without checking out the rest of the AgCrawler / CyberVisionAg repos.

For SSH details, cluster paths, transfer/resync commands, and monitoring
recipes specific to ISU Nova, see [CLUSTER.md](CLUSTER.md).

What it does: walks a per-crop directory of labeled images, asks Claude
(via the Anthropic API) whether each image matches the symptom KB for
its class, and copies matched images into part-tagged reference and
test splits.

## Layout

```
capsuled_data_prep/
├── prepare_dataset.py     # entry script (self-contained)
├── requirements.txt       # anthropic, openpyxl, Pillow
├── setup.sh               # creates .venv, installs deps, prompts for key
├── run_example.sh         # example invocation
├── .env.example           # API-key template
└── kb/                    # KB workbooks shipped with the capsule
    ├── Banana/internet.xlsx
    ├── Cauliflower/internet.xlsx
    ├── Coffee/internet.xlsx
    ├── Corn/internet.xlsx
    ├── Mango_Leaf/internet.xlsx
    ├── Orange/internet.xlsx
    ├── Soybean/internet.xlsx
    ├── Sugarcane/internet.xlsx
    ├── Tomato/internet.xlsx
    └── Wheat/internet.xlsx
```

The KB folder name must match the crop prefix derived from your input
directory name. The script strips trailing `_Diseases` / `_Disease`, so
`Tomato_Diseases` looks up `kb/Tomato/internet.xlsx`, and
`Mango_Leaf_Disease` looks up `kb/Mango_Leaf/internet.xlsx`.

## Quick start (on the cluster)

1. Copy the entire `capsuled_data_prep/` folder to the cluster.
2. Run setup once:
   ```bash
   cd capsuled_data_prep
   bash setup.sh
   ```
   `setup.sh` creates `.venv/`, installs the three Python deps, prompts
   for your `ANTHROPIC_API_KEY` (or picks it up from the environment),
   writes it to `.env`, and runs a one-token smoke test against the API.
3. Activate the venv and run a dataset:
   ```bash
   source .venv/bin/activate
   python prepare_dataset.py \
     --input-dir /work/mech-ai-scratch/tirtho/CyAg/Curated_Dataset/Images/Tomato_Diseases \
     --output-dir ./out/Tomato_Diseases \
     --max-per-part 5 \
     --test-per-class 5 \
     --max-inspect-per-class 60 \
     --seed 42 \
     --parallel 12
   ```
   Or use the prepared driver:
   ```bash
   CROP=Tomato bash run_example.sh
   ```

## Outputs

For input dir `<DATASET_ROOT>/<Crop>_Diseases/<class>/<image>.jpg`:

```
out/<Crop>_Diseases/
├── <class>/
│   ├── leaf/        # up to --max-per-part matched images, by part
│   ├── stem/
│   ├── ...
│   └── rejected/    # images that did not match the KB
└── _tags.csv        # audit log: class, file, match, part, split, reason

out/<Crop>_Diseases_test/   # separate root, for --test-per-class images
└── <class>/
```

## Arguments

| flag | default | what it does |
|---|---|---|
| `--input-dir` | required | source folder; expects `<class>/<image>.{jpg,jpeg,png,webp}` |
| `--output-dir` | required | reference output root; test split goes to `<output-dir>_test` |
| `--max-per-part` | 5 | reference cap per (class, plant_part) |
| `--test-per-class` | 0 | test cap per class (0 = no test split) |
| `--max-inspect-per-class` | all | hard cap on images to inspect per class |
| `--seed` | 42 | shuffle seed |
| `--parallel` | 1 | concurrent API calls |
| `--filename-prefix` | none | only process files starting with this prefix |
| `--exclude` | none | comma-separated class names to skip |

## Cost / time guidance

Each image = one Claude Sonnet call with one image attachment, up to
~200 output tokens. Roughly 30–50 images per minute at `--parallel 12`
on a stable network. Cost order of magnitude: ~\$0.005–0.015 per image
depending on image size.

If you want a quick directional pass before committing to the full
sweep, set `--max-inspect-per-class 30 --max-per-part 3 --test-per-class 3`.

## Troubleshooting

* **`ANTHROPIC_API_KEY not found`** — re-run `bash setup.sh` or export
  the variable before invoking the script.
* **`KB not found at kb/<Crop>/internet.xlsx`** — the crop name parsed
  from `--input-dir` does not match a folder in `kb/`. Rename the
  folder under `kb/` or rename the input directory's leaf segment.
* **HTTP 429 / overloaded** — drop `--parallel` to 4–6.
* **Pillow not installed** — only required for >3.7 MB source images;
  `pip install Pillow` from inside `.venv` if needed.
