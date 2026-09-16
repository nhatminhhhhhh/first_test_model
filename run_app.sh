#!/bin/bash
# Launches app.py. Meant to be called by the desktop shortcut created via
# scripts/install_shortcut.sh, but works standalone too:
#   ./run_app.sh
# Uses this repo's .venv if present, otherwise falls back to system python3
# (e.g. when dependencies were installed system-wide with apt/pip instead).
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    python app.py
else
    python3 app.py
fi
