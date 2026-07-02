#!/usr/bin/env bash
set -euo pipefail

# Legacy script name kept for existing workflows. The Studio backend now uses
# Antfly's unified inference binary.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$REPO_ROOT/antfly/zig"

cd "$SRC_DIR"
zig build install "$@"

BUILT="$SRC_DIR/zig-out/bin/antfly"
INSTALL_DIR="${ANTFLY_BIN_DIR:-$HOME/.unsloth/antfly/bin}"
OUT_BIN="$INSTALL_DIR/antfly"
mkdir -p "$INSTALL_DIR"
install -m 755 "$BUILT" "$OUT_BIN"

# macOS 26's code-signing monitor rejects Zig's linker-emitted ad-hoc
# signature at exec ("Taskgated Invalid Signature"). Re-sign with the
# system codesign tool.
if [[ "$(uname -s)" == "Darwin" ]]; then
    codesign --force --sign - "$OUT_BIN"
fi

echo "✅ $OUT_BIN"
