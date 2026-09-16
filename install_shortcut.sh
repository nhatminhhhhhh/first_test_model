#!/bin/bash
# Creates a double-clickable "Drowning Detection" shortcut on the Raspberry Pi
# desktop and in the applications menu, pointing at this repo's run_app.sh.
# Run once on the Pi itself (paths are auto-detected from where this repo lives):
#   bash scripts/install_shortcut.sh
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="$REPO_DIR/run_app.sh"
chmod +x "$LAUNCHER"

DESKTOP_ENTRY="[Desktop Entry]
Type=Application
Name=Drowning Detection
Comment=Chay giao dien phat hien duoi nuoc
Exec=$LAUNCHER
Path=$REPO_DIR
Icon=camera-web
Terminal=true
Categories=Utility;
"

mkdir -p "$HOME/Desktop" "$HOME/.local/share/applications"

for target in "$HOME/Desktop/drowning-detection.desktop" "$HOME/.local/share/applications/drowning-detection.desktop"; do
    printf '%s' "$DESKTOP_ENTRY" > "$target"
    chmod +x "$target"
    if command -v gio >/dev/null 2>&1; then
        gio set "$target" "metadata::trusted" true 2>/dev/null || true
    fi
done

echo "Da tao shortcut 'Drowning Detection' tren Desktop va trong menu ung dung (Categories: Utility)."
echo "Neu icon tren Desktop van bao 'untrusted', chuot phai vao icon > Trust/Allow Launching / Properties > Permissions > Execute."
