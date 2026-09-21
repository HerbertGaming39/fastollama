#!/usr/bin/env bash
# fastollama uninstaller — removes what install.sh created.
set -euo pipefail

PREFIX="${PREFIX:-$HOME/.local/share/fastollama}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"

echo "removing fastollama from $PREFIX"
rm -rf "$PREFIX"
rm -f "$BIN_DIR/fastollama"
echo "done. (downloaded models lived inside $PREFIX/models — they are gone too)"
