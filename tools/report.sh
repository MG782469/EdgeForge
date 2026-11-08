#!/usr/bin/env bash
# EdgeForge - Stage 8: one reproducible benchmark report for this machine.
#
#   bash tools/report.sh MODEL_DIR EFQ [OUT.md]
#
# MODEL_DIR : the original BF16 TinyLlama folder; EFQ: the model from tools/quantize_tinyllama.sh
# OUT.md    : default bench/report_<hostname>.md (plain text, safe to commit)
# Environment: EF_BIN (~/ef-build/edgeforge), SKIP_MEMBENCH=1, SKIP_BF16=1 (skip the slow BF16 ladder),
#              SKIP_CHAT=1 (skip verify-chat, which runs when tests/golden/chat_generation.json exists)
#
# Runs, from the repository root, every measurement the README quotes and stores each command
# with its full output: versions, sysinfo, membench, verify-quant, eval-quant, the bench-decode
# speed-up ladder (4-bit and BF16), the per-op profile and verify-chat. About 10 minutes on the demo
# laptop, plus about 15 for verify-chat.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR=$(cd "$1" && pwd)
EFQ=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
OUT_MD=${3:-$ROOT/bench/report_$(hostname).md}
EF=${EF_BIN:-$HOME/ef-build/edgeforge}
[[ -x "$EF" ]] || { echo "edgeforge binary not found at $EF (set EF_BIN)" >&2; exit 2; }
[[ -f "$EFQ" ]] || { echo "no .efq file at $EFQ" >&2; exit 2; }
cd "$ROOT"
mkdir -p "$(dirname "$OUT_MD")"

section() {  # section TITLE COMMAND...: run it, keep the command line and all of its output
  local title=$1
  shift
  echo "== $title"
  {
    echo
    echo "## $title"
    echo
    echo '```'
    echo "\$ $*"
    "$@" 2>&1 || echo "(exit code $?)"
    echo '```'
  } >> "$OUT_MD"
}

dirty=$(git status --porcelain 2>/dev/null | grep -v '^??' | wc -l || true)
{
  echo "# EdgeForge benchmark report: $(hostname)"
  echo
  echo '```'
  echo "date      : $(date -Iseconds)"
  echo "edgeforge : $("$EF" --version), git $(git rev-parse --short HEAD 2>/dev/null || echo unknown)$([[ "$dirty" != 0 ]] && echo ' (uncommitted changes)')"
  echo "model     : $MODEL_DIR"
  echo "efq       : $EFQ ($(du -h "$EFQ" | cut -f1)), sha256 $(sha256sum "$EFQ" | cut -c1-16)..."
  echo '```'
} > "$OUT_MD"

section "System" "$EF" sysinfo
if [[ "${SKIP_MEMBENCH:-0}" != 1 ]]; then
  section "Memory read bandwidth (membench)" "$EF" membench
fi
section "Quantized file vs the BF16 checkpoint (verify-quant)" \
  "$EF" verify-quant --qmodel "$EFQ" --model-dir "$MODEL_DIR" --golden tests/golden
section "Quality and speed of the quantized engine (eval-quant)" \
  "$EF" eval-quant --qmodel "$EFQ" --model-dir "$MODEL_DIR" --golden tests/golden --text tests/fixtures/eval_text.txt
section "Speed-up ladder, 4-bit engine (bench-decode)" "$EF" bench-decode --qmodel "$EFQ"
if [[ "${SKIP_BF16:-0}" != 1 ]]; then
  section "Speed-up ladder, BF16 engine (bench-decode)" "$EF" bench-decode --model-dir "$MODEL_DIR" --tokens 16
fi
section "Where the time goes (profile)" "$EF" profile --qmodel "$EFQ"
if [[ "${SKIP_CHAT:-0}" != 1 && -f tests/golden/chat_generation.json ]]; then
  section "Chat replies vs Hugging Face, BF16 engine (verify-chat)" \
    "$EF" verify-chat --golden tests/golden --model-dir "$MODEL_DIR"
  section "Chat replies vs Hugging Face, 4-bit engine (verify-chat)" \
    "$EF" verify-chat --golden tests/golden --qmodel "$EFQ" --tokenizer-dir "$MODEL_DIR"
fi
echo "wrote $OUT_MD"
