#!/usr/bin/env bash
# EdgeForge Studio - start the local bridge + web UI (WSL / Linux).
#
#   bash studio/run.sh [MODEL_DIR] [EFQ]      then open http://localhost:8790 in the Windows browser
#
# Defaults: MODEL_DIR ~/edgeforge-data/tinyllama, EFQ ~/edgeforge-data/tinyllama-q4.efq,
#           EF_BIN ~/ef-build/edgeforge, PORT 8790. Needs Node.js 18+ (no npm install: the web app in
#           studio/web/dist is prebuilt and the bridge uses Node built-ins only).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_DIR=${1:-$HOME/edgeforge-data/tinyllama}
EFQ=${2:-$HOME/edgeforge-data/tinyllama-q4.efq}
EF=${EF_BIN:-$HOME/ef-build/edgeforge}
PORT=${PORT:-8790}
HOST=${HOST:-127.0.0.1}   # HOST=0.0.0.0 only if the Windows browser cannot reach WSL's localhost

command -v node > /dev/null || { echo "Node.js not found: sudo apt install -y nodejs" >&2; exit 2; }
major=$(node -p 'process.versions.node.split(".")[0]')
[[ $major -ge 18 ]] || { echo "Node.js $major is too old (need 18+)" >&2; exit 2; }
[[ -x "$EF" ]] || { echo "edgeforge binary not found at $EF (build it, or set EF_BIN)" >&2; exit 2; }
[[ -f "$EFQ" ]] || echo "warning: no .efq at $EFQ (the Q4 model will not start)" >&2
[[ -f "$MODEL_DIR/tokenizer.json" ]] || { echo "no tokenizer.json in $MODEL_DIR" >&2; exit 2; }
[[ -f "$HERE/web/dist/app.js" ]] || { echo "studio/web/dist missing: cd studio/web && npm install && node build.mjs" >&2; exit 2; }
exec node "$HERE/server.mjs" --bin "$EF" --model-dir "$MODEL_DIR" --efq "$EFQ" --port "$PORT" --host "$HOST"
