#!/usr/bin/env bash
# Builds dist/fastollama-v<version>-linux-x86_64.tar.gz
# Layout (matches install.sh and the app's own path resolution):
#   fastollama-v*/
#     bin/fastollama           (governor binary; expects settings at ../settings.txt)
#     engine/llama-server      (vulkan llama.cpp server)
#     settings.txt
#     install.sh  uninstall.sh  README.md  LICENSE
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${1:-$(git describe --tags --abbrev=0 2>/dev/null || echo v0.0-dev)}"
NAME="fastollama-${VERSION}-linux-x86_64"
STAGE="dist/$NAME"

[ -x bin/fastollama ]              || { echo "missing bin/fastollama — build it first";              exit 1; }
[ -x llama.cpp/build-vk/bin/llama-server ] || { echo "missing llama.cpp/build-vk/bin/llama-server — build the vulkan backend first"; exit 1; }

rm -rf "$STAGE" && mkdir -p "$STAGE/bin" "$STAGE/engine"
cp bin/fastollama                        "$STAGE/bin/fastollama"
cp llama.cpp/build-vk/bin/llama-server   "$STAGE/engine/llama-server"
cp settings.txt                          "$STAGE/settings.txt"
cp install.sh uninstall.sh README.md LICENSE "$STAGE/"

tar -C dist -czf "dist/$NAME.tar.gz" "$NAME"
rm -rf "$STAGE"
echo "built dist/$NAME.tar.gz:"
ls -lh "dist/$NAME.tar.gz"
