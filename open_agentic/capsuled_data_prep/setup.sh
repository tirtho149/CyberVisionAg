#!/usr/bin/env bash
# One-shot setup: creates a venv, installs deps, prompts for API key.
# Run from this directory:  bash setup.sh
set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python}"
if ! command -v "${PY}" >/dev/null 2>&1; then
    PY="python3"
fi
echo "Using interpreter: $(${PY} -V)"

# 1. venv
if [ ! -d ".venv" ]; then
    echo "Creating virtualenv at .venv ..."
    "${PY}" -m venv .venv
else
    echo "Reusing existing .venv"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt
echo "Installed: $(python -m pip list 2>/dev/null | grep -E 'anthropic|openpyxl|Pillow' | tr '\n' ' ')"

# 2. API key
if [ -f ".env" ] && grep -q "ANTHROPIC_API_KEY=" .env 2>/dev/null; then
    echo ".env already contains ANTHROPIC_API_KEY (leaving untouched)."
else
    if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
        echo "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}" > .env
        echo "Wrote .env from environment."
    else
        echo
        echo "Paste your ANTHROPIC_API_KEY (input is hidden):"
        read -rs KEY
        echo "ANTHROPIC_API_KEY=${KEY}" > .env
        chmod 600 .env
        echo "Wrote .env (chmod 600)."
    fi
fi

# 3. quick smoke
echo
echo "Smoke-test API access ..."
python - <<'PY'
import os, sys
from pathlib import Path
env_path = Path(__file__).resolve().parent / ".env" if "__file__" in dir() else Path(".env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
            break
import anthropic
client = anthropic.Anthropic()
resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=10,
    messages=[{"role": "user", "content": "Reply OK"}],
)
print("API OK:", resp.content[0].text.strip())
PY

echo
echo "Setup complete. Activate the venv before running:"
echo "  source .venv/bin/activate"
echo "Then see run_example.sh for an example invocation."
