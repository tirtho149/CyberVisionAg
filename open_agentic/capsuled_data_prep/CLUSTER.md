# Cluster operations runbook

Operating notes for running `prepare_dataset` on Iowa State's Nova HPC
cluster. The capsule itself is documented in [README.md](README.md);
this file covers the SSH / file-sync / monitoring flow around it.

## Identifiers

| field | value |
|---|---|
| cluster host (data transfer node) | `novadtn.its.iastate.edu` |
| compute login (interactive nodes) | `nova.its.iastate.edu` |
| user | `arbab` |
| group scratch | `/work/mech-ai-scratch/` |
| capsule home on cluster | `/work/mech-ai-scratch/arbab/sage/` |
| source dataset on cluster | `/work/mech-ai-scratch/tirtho/CyAg/Curated_Dataset/Images/` |
| local capsule source of truth | `CyberVisionAg/open_agentic/capsuled_data_prep/` |

ISU's docs require all bulk transfers to go through `novadtn` (the data
transfer node). Login nodes work for scripting; large copies should use
`novadtn`.

## SSH

Key auth is already configured for `arbab@novadtn.its.iastate.edu`. If
you ever see a host-key warning after a cluster rekey:

```bash
ssh-keygen -R novadtn.its.iastate.edu
ssh arbab@novadtn.its.iastate.edu      # accept the new fingerprint once
```

Quick sanity check:

```bash
ssh arbab@novadtn.its.iastate.edu 'whoami; hostname; ls /work/mech-ai-scratch/arbab/sage/'
```

The harmless `module: command not found` lines in the output come from
`~/.bashrc` — Lmod isn't initialized in non-interactive `ssh "<cmd>"`.
Ignore them.

## Initial transfer of the capsule

From your local machine (one shot, on first ship and after major
edits):

```bash
ssh arbab@novadtn.its.iastate.edu "mkdir -p /work/mech-ai-scratch/arbab/sage" && \
scp -r /Users/muhammadarbabarshad/build2026-local/AgCrawler/CyberVisionAg/open_agentic/capsuled_data_prep/. \
       arbab@novadtn.its.iastate.edu:/work/mech-ai-scratch/arbab/sage/
```

The trailing `.` is intentional: it copies the **contents** of
`capsuled_data_prep/` into `sage/` so dotfiles like `.env.example` come
along.

`rsync` equivalent (resumable, faster on slow links):

```bash
rsync -avz --progress \
  /Users/muhammadarbabarshad/build2026-local/AgCrawler/CyberVisionAg/open_agentic/capsuled_data_prep/ \
  arbab@novadtn.its.iastate.edu:/work/mech-ai-scratch/arbab/sage/
```

## Resyncing after a code change

Use this when you edit `prepare_dataset.py`, the KB, or the helper
scripts locally and need to push the change to the cluster without
re-uploading the venv or output directories:

```bash
rsync -avz --progress \
  --exclude='.venv/' --exclude='out/' --exclude='logs/' --exclude='.env' \
  /Users/muhammadarbabarshad/build2026-local/AgCrawler/CyberVisionAg/open_agentic/capsuled_data_prep/ \
  arbab@novadtn.its.iastate.edu:/work/mech-ai-scratch/arbab/sage/
```

For a single-file fix (e.g. when adding `from __future__ import annotations`):

```bash
scp /Users/muhammadarbabarshad/build2026-local/AgCrawler/CyberVisionAg/open_agentic/capsuled_data_prep/prepare_dataset.py \
    arbab@novadtn.its.iastate.edu:/work/mech-ai-scratch/arbab/sage/prepare_dataset.py
```

If you want the cluster to pull straight from the SAGE GitHub repo
instead of from your laptop, that works too:

```bash
ssh arbab@novadtn.its.iastate.edu '
  cd /work/mech-ai-scratch/arbab/sage && \
  curl -fsSL -o prepare_dataset.py \
    https://raw.githubusercontent.com/tirtho149/SAGE/main/open_agentic/capsuled_data_prep/prepare_dataset.py
'
```

## Refreshing the KB

When the disease registry changes locally and you want the new KB on
the cluster:

```bash
# from CyberVisionAg/, refresh capsule's bundled KB from the registry outputs
for c in Banana Cauliflower Coffee Corn Mango_Leaf Orange Soybean Sugarcane Tomato Wheat; do
  cp disease_registry/outputs/$c/internet.xlsx \
     open_agentic/capsuled_data_prep/kb/$c/
done

# then ship just kb/ to the cluster
rsync -avz --progress \
  open_agentic/capsuled_data_prep/kb/ \
  arbab@novadtn.its.iastate.edu:/work/mech-ai-scratch/arbab/sage/kb/
```

## One-time setup on the cluster

```bash
ssh arbab@novadtn.its.iastate.edu
cd /work/mech-ai-scratch/arbab/sage
bash setup.sh
```

`setup.sh` creates `.venv/`, installs `anthropic` + `openpyxl` + `Pillow`,
prompts for `ANTHROPIC_API_KEY` (input hidden), writes `.env` with
`chmod 600`, and runs a one-token API smoke test. Idempotent — running
it again reuses the venv and will refuse to clobber a working `.env`.

The cluster has Python 3.9 system-wide; the script is annotated with
`from __future__ import annotations` so it works without needing a
newer interpreter or module-loaded Python build.

## Running prepare_dataset on the cluster

Foreground (you watch it run, lose it if your laptop sleeps):

```bash
ssh arbab@novadtn.its.iastate.edu
cd /work/mech-ai-scratch/arbab/sage
source .venv/bin/activate
CROP=Tomato bash run_example.sh
```

Background (recommended; survives disconnects):

```bash
ssh arbab@novadtn.its.iastate.edu '
  cd /work/mech-ai-scratch/arbab/sage && \
  mkdir -p logs out && \
  source .venv/bin/activate && \
  nohup python prepare_dataset.py \
    --input-dir /work/mech-ai-scratch/tirtho/CyAg/Curated_Dataset/Images/Tomato \
    --output-dir ./out/Tomato \
    --max-per-part 5 --test-per-class 5 \
    --max-inspect-per-class 60 --seed 42 --parallel 12 \
    > logs/Tomato.log 2>&1 < /dev/null & \
  echo "PID=$!"'
```

Per-crop input dir naming on the cluster (no `_Diseases` suffix, except
where the source dataset is named that way):

| crop | input dir |
|---|---|
| Soybean | `/work/mech-ai-scratch/tirtho/CyAg/Curated_Dataset/Images/Soybean` |
| Corn | `.../Images/Corn` |
| Mango | `.../Images/Mango` |
| Tomato | `.../Images/Tomato` |
| Banana | `.../Images/Banana` |
| Cauliflower | `.../Images/Cauliflower` |
| Coffee | `.../Images/Coffee` |
| Orange | `.../Images/Orange` |
| Sugarcane | `.../Images/Sugarcane` |
| Wheat | `.../Images/Wheat` |

The capsule's `load_kb()` strips `_Diseases` / `_Disease` from the
input dir's leaf name. So `Tomato` resolves to `kb/Tomato/`. For Mango,
the dir is `Mango` on the cluster but the registry KB folder is
`kb/Mango_Leaf/` — if you run Mango, either rename the input or copy
`kb/Mango_Leaf/internet.xlsx` to `kb/Mango/internet.xlsx` first.

## Monitoring a running job

```bash
# Process status
ssh arbab@novadtn.its.iastate.edu 'pgrep -af prepare_dataset.py | head -3'

# Files produced so far (responsive progress signal — log buffers slowly)
ssh arbab@novadtn.its.iastate.edu \
  'find /work/mech-ai-scratch/arbab/sage/out -type f 2>/dev/null | wc -l'

# Log tail (only flushes on class boundaries due to Python buffering)
ssh arbab@novadtn.its.iastate.edu 'tail -30 /work/mech-ai-scratch/arbab/sage/logs/Tomato.log'

# Done check
ssh arbab@novadtn.its.iastate.edu \
  'test -f /work/mech-ai-scratch/arbab/sage/out/Tomato/_tags.csv && echo DONE || echo running'
```

If you'd rather have unbuffered Python output (for live `tail -f`
watching), launch with `python -u prepare_dataset.py ...`.

## Pulling results back to your local machine

```bash
rsync -avz --progress \
  arbab@novadtn.its.iastate.edu:'/work/mech-ai-scratch/arbab/sage/out/Tomato /work/mech-ai-scratch/arbab/sage/out/Tomato_test' \
  ~/build2/AgCrawler/CyberVisionAg/Prepared_Dataset/
```

(Note: with `rsync`, the multi-source quoting trick on the right of `:`
copies both directories in one transfer.)

The audit log lives at `out/<Crop>/_tags.csv` and is small — sometimes
worth pulling separately to inspect before downloading the images:

```bash
scp arbab@novadtn.its.iastate.edu:/work/mech-ai-scratch/arbab/sage/out/Tomato/_tags.csv ./
```

## Killing a runaway job

```bash
ssh arbab@novadtn.its.iastate.edu 'pkill -f "prepare_dataset.py.*Tomato"'
# or, with surgical PID:
ssh arbab@novadtn.its.iastate.edu 'kill <PID>'
```

## Known gotchas

* **Python 3.9 only**: cluster has 3.9.21. The script uses
  `from __future__ import annotations` so PEP 604 (`str | None`) is
  fine in annotations, but don't add 3.10+-only runtime constructs
  (e.g. `match` statements, parenthesized context managers).
* **`module: command not found`**: harmless. Lmod isn't loaded in
  non-interactive `ssh "<cmd>"` invocations. Doesn't affect anything.
* **Empty log file mid-run**: stdout is line-buffered; `prepare_dataset`
  only prints per-class. Use the file count under `out/` as a live
  progress indicator instead.
* **`ANTHROPIC_API_KEY not found`**: re-run `bash setup.sh` or
  `export ANTHROPIC_API_KEY=...` before launching python.
* **HTTP 429 / overloaded**: drop `--parallel` from 12 to 4–6.
* **Storage**: `/work/mech-ai-scratch/` is scratch and not backed up.
  Pull final outputs back to your laptop or to a longer-term archive.
* **Compute on data transfer node**: `novadtn` is fine for I/O and
  short scripts. For long sweeps (hours), submit a Slurm job from a
  login node instead so you don't get killed for using DTN CPU.

## Quick reference cheat sheet

```bash
# SSH in
ssh arbab@novadtn.its.iastate.edu

# Where to be
cd /work/mech-ai-scratch/arbab/sage
source .venv/bin/activate

# Run (background)
nohup python prepare_dataset.py --input-dir /work/mech-ai-scratch/tirtho/CyAg/Curated_Dataset/Images/<Crop> \
  --output-dir ./out/<Crop> --max-per-part 5 --test-per-class 5 \
  --max-inspect-per-class 60 --seed 42 --parallel 12 \
  > logs/<Crop>.log 2>&1 < /dev/null &

# Watch
watch -n 5 'find out -type f | wc -l; tail -3 logs/<Crop>.log'

# Pull results back (from local)
rsync -avz arbab@novadtn.its.iastate.edu:/work/mech-ai-scratch/arbab/sage/out/<Crop> \
            ~/build2/AgCrawler/CyberVisionAg/Prepared_Dataset/
```
