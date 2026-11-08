#!/usr/bin/env bash
# EdgeForge - Stage 8: the live demo (about 3 minutes, then an open chat).
#
#   bash tools/demo.sh MODEL_DIR EFQ
#
# Shows: the model on disk (2.05 GiB -> 713 MiB), this CPU, one answer from the 4-bit engine with
# its speed, the speed-up ladder (scalar 1 thread -> AVX2 + threads, outputs checked), then an
# interactive chat. Environment: EF_BIN (~/ef-build/edgeforge), THREADS (physical cores),
# PAUSE=0 (do not wait for Enter between steps).
set -euo pipefail

if [[ $# -lt 2 ]]; then
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR=$(cd "$1" && pwd)
EFQ=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
EF=${EF_BIN:-$HOME/ef-build/edgeforge}
[[ -x "$EF" ]] || { echo "edgeforge binary not found at $EF (set EF_BIN)" >&2; exit 2; }
THREADS_ARG=()
[[ -n "${THREADS:-}" ]] && THREADS_ARG=(--threads "$THREADS")
cd "$ROOT"

step() {
  echo
  echo "================================================================================"
  echo "  $1"
  echo "================================================================================"
}
pause() {
  if [[ "${PAUSE:-1}" != 0 ]]; then
    read -r -p "  [Enter] to continue " _
  fi
}

step "1. The same TinyLlama-1.1B, before and after our 4-bit quantizer"
for f in "$MODEL_DIR/model.safetensors" "$EFQ"; do
  ls -lh "$f" | awk '{print "   ", $5, $9}'
done
pause

step "2. This laptop"
"$EF" sysinfo | head -n 12
pause

step "3. One answer from the 4-bit engine (no ML libraries, our own C++ kernels)"
"$EF" chat --qmodel "$EFQ" --tokenizer-dir "$MODEL_DIR" ${THREADS_ARG[@]+"${THREADS_ARG[@]}"} \
  --prompt "What is the capital of France? Answer in one sentence."
pause

step "4. Speed-up ladder: scalar 1 thread -> AVX2 -> AVX2 + threads (same outputs checked)"
"$EF" bench-decode --qmodel "$EFQ" --tokens 32
pause

step "5. Your turn: chat with the 4-bit model (/reset clears, /exit quits)"
"$EF" chat --qmodel "$EFQ" --tokenizer-dir "$MODEL_DIR" ${THREADS_ARG[@]+"${THREADS_ARG[@]}"}
