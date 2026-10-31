#!/usr/bin/env bash
# EdgeForge - reproduce the verified Stage 5 model from the original BF16 checkpoint.
#
#   bash tools/quantize_tinyllama.sh MODEL_DIR OUT.efq [GOLDEN_DIR]
#
# MODEL_DIR : TinyLlama/TinyLlama-1.1B-Chat-v1.0 (config.json + model.safetensors, BF16)
# OUT.efq   : output file (keep it outside the repo/OneDrive; *.efq is git-ignored)
# GOLDEN_DIR: default tests/golden
# EF_BIN    : edgeforge binary, default ~/ef-build/edgeforge
#
# Quantizes with the named, version-controlled policy (Q4_0 layers, 6 measured-sensitive matrices
# in Q8_0, Q8_0 lm_head, BF16 embedding, F32 norms), then runs the full verification. Exit code 0
# only if verify-quant passes. The output is byte-identical for any thread count, so the printed
# sha256 must be the same on every run with the same checkpoint and binary.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
MODEL_DIR=$1
OUT=$2
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GOLDEN=${3:-$ROOT/tests/golden}
EF=${EF_BIN:-$HOME/ef-build/edgeforge}

[[ -x "$EF" ]] || { echo "edgeforge binary not found at $EF (set EF_BIN)" >&2; exit 2; }
[[ -f "$MODEL_DIR/model.safetensors" ]] || { echo "no model.safetensors in $MODEL_DIR" >&2; exit 2; }

echo "== quantize (policy tinyllama-1.1b-chat-v1.0)"
"$EF" quantize --model-dir "$MODEL_DIR" --out "$OUT" --policy tinyllama-1.1b-chat-v1.0

echo
echo "== verify"
"$EF" verify-quant --qmodel "$OUT" --model-dir "$MODEL_DIR" --golden "$GOLDEN"

echo
if command -v sha256sum >/dev/null; then
  echo "sha256: $(sha256sum "$OUT" | cut -d' ' -f1)  $(basename "$OUT")"
fi
echo "reproduced and verified: $OUT"
