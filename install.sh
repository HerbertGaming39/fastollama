#!/usr/bin/env bash
# fastollama installer — no root needed by default.
# Usage:
#   ./install.sh                                -> install to ~/.local/share/fastollama
#   PREFIX=/opt/fastollama sudo ./install.sh    -> system-wide install
set -euo pipefail

C_GREEN='\033[0;32m'; C_RED='\033[0;31m'; C_YELL='\033[0;33m'; C_OFF='\033[0m'
ok()   { printf "  ${C_GREEN}✔${C_OFF} %s\n" "$1"; }
warn() { printf "  ${C_YELL}!${C_OFF} %s\n" "$1"; }
fail() { printf "  ${C_RED}✘ %s${C_OFF}\n" "$1"; exit 1; }

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local/share/fastollama}"

echo "fastollama installer"
echo "  source: $SRC_DIR"
echo "  target: $PREFIX"
echo

# --- checks -------------------------------------------------------------
[ -x "$SRC_DIR/engine/llama-server" ] || fail "engine/llama-server not found next to install.sh (use the release tarball)"
[ -x "$SRC_DIR/bin/fastollama" ]      || fail "bin/fastollama not found next to install.sh (use the release tarball)"
[ -f "$SRC_DIR/settings.txt" ]        || fail "settings.txt not found next to install.sh"

for lib in vulkan; do
  # no grep -q here: with pipefail it SIGPIPEs the producer and false-negatives
  ldconfig -p 2>/dev/null | grep "lib${lib}.so" >/dev/null || fail "libvulkan not found — install your GPU driver's Vulkan runtime first (mesa-vulkan-drivers / vulkan-radeon)"
done
ok "libvulkan found"

# --- install -------------------------------------------------------------
# app convention: binary lives in <base>/bin, settings in <base>/, models in <base>/models
mkdir -p "$PREFIX/bin" "$PREFIX/engine" "$PREFIX/models" "$PREFIX/logs"
install -m 0755 "$SRC_DIR/engine/llama-server" "$PREFIX/engine/llama-server"
install -m 0755 "$SRC_DIR/bin/fastollama"     "$PREFIX/bin/fastollama"
install -m 0644 "$SRC_DIR/settings.txt"        "$PREFIX/settings.txt"
ok "binaries + settings copied"

# point the app at the flattened engine layout
grep -q '^engine_dir' "$PREFIX/settings.txt" \
  && sed -i 's|^engine_dir.*|engine_dir = engine|' "$PREFIX/settings.txt" \
  || printf '\n# installed layout: engine lives next to settings.txt\nengine_dir = engine\n' >> "$PREFIX/settings.txt"
ok "settings.txt wired (engine_dir = engine)"

# --- launcher on PATH ---------------------------------------------------
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
mkdir -p "$BIN_DIR"
ln -sf "$PREFIX/bin/fastollama" "$BIN_DIR/fastollama"
ok "launcher symlinked: $BIN_DIR/fastollama"

case ":$PATH:" in
  *":$BIN_DIR:"*) ok "$BIN_DIR is on PATH" ;;
  *) warn "$BIN_DIR is NOT on PATH — add it:  echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.bashrc" ;;
esac

echo
echo "done. get a model and run:"
echo "  mkdir -p $PREFIX/models"
echo "  curl -L -o $PREFIX/models/Qwen3.8-27B-UD-IQ2_S.gguf <model-url>"
echo "  fastollama serve"
