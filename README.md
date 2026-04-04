# CyberVisionAg

Agentic plant disease classification using Claude Code (`claude -p`) in subprocess mode. Each test image gets its own autonomous Claude agent that reasons freely, views reference images, and consults a symptom knowledge base.

> The old `agent.py` fixed-pipeline is superseded by `open_agentic/`. See below for pointers.

## Getting Started

**Prerequisites**: conda env `vl-reasoning` (Python 3.10), Claude Code CLI installed and authenticated, `.env` with `ANTHROPIC_API_KEY` in the AgCrawler root.

- **Setup, architecture, run commands, and adding new crops**: see [open_agentic/README.md](open_agentic/README.md)
- **Knowledge base generation** (symptom extraction from PDFs and web): see [disease_registry/README.md](../disease_registry/README.md)
- **Paper writing plan**: see [open_agentic/storyline.md](open_agentic/storyline.md)

## Quick Start

```bash
cd /Users/muhammadarbabarshad/build2026-local/AgCrawler
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vl-reasoning
set -a && source .env && set +a

# Smoke test (2 classes, 1 image each)
python -m CyberVisionAg.open_agentic.eval --symptom-source local --quick-test 2
```
