#!/bin/bash
# Launches app.py using this repo's venv. Meant to be called by the desktop
# shortcut created via scripts/install_shortcut.sh, but works standalone too:
#   ./run_app.sh
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
source .venv/bin/activate
python app.py
