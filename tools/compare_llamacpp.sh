#!/usr/bin/env bash
# EdgeForge - Stage 8: EdgeForge vs llama.cpp on the same laptop, model, text and thread count.
#
#   bash tools/compare_llamacpp.sh LLAMA_CPP_DIR MODEL_DIR EFQ [OUT_DIR]
#
# LLAMA_CPP_DIR : a llama.cpp checkout built with CMake (build/bin/llama-bench, llama-perplexity,
#                 llama-quantize); PYTHON must have the packages of its convert_hf_to_gguf.py
# MODEL_DIR     : the original BF16 TinyLlama-1.1B-Chat-v1.0 folder (config.json, model.safetensors,
#                 tokenizer files)
# EFQ           : the EdgeForge model from tools/quantize_tinyllama.sh
# OUT_DIR       : logs, GGUF files and summary.md (default ~/edgeforge-data/compare-YYYYmmdd-HHMM)
# Environment   : THREADS (6), REPS (5), CTX (512), CHUNKS (8), TEXT (default: the wikitext-2 raw test
#                 set from llama.cpp's scripts/get-wikitext-2.sh), EF_BIN (~/ef-build/edgeforge),
#                 PYTHON (python3)
#
# Matched settings:
#   - weights: llama.cpp converts the same BF16 checkpoint to a BF16 GGUF (exact copy), then to its own
#     Q4_0 recipe with llama-quantize. EdgeForge uses its .efq (Q4_0 + 6 Q8_0 matrices, Q8_0 lm_head).
#   - speed: same threads; 32 decode steps after the 23-token golden prompt p0_factual (llama-bench
#     tg32 @ d23 vs EdgeForge bench-decode), REPS runs each. Prompt processing is NOT compared: llama.cpp batches
#     the prompt (matrix-matrix), EdgeForge runs it one token at a time (matrix-vector).
#   - quality: same text, same token ids (checked), same chunking as llama-perplexity (-c CTX
#     --chunks CHUNKS: BOS first, second half of each chunk scored). Each engine's quantized model is
#     compared with its own BF16 model: perplexity, mean KL divergence and top-1 agreement.
set -euo pipefail

if [[ $# -lt 3 ]]; then
  sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
LLAMA=$(cd "$1" && pwd)
MODEL_DIR=$(cd "$2" && pwd)
EFQ=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
OUT=${4:-$HOME/edgeforge-data/compare-$(date +%Y%m%d-%H%M)}
THREADS=${THREADS:-6}
REPS=${REPS:-5}
CTX=${CTX:-512}
CHUNKS=${CHUNKS:-8}
EF=${EF_BIN:-$HOME/ef-build/edgeforge}
PY=${PYTHON:-python3}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=$LLAMA/build/bin

for b in llama-bench llama-perplexity llama-quantize; do
  [[ -x "$BIN/$b" ]] || { echo "missing $BIN/$b (build llama.cpp with CMake first)" >&2; exit 2; }
done
[[ -x "$EF" ]] || { echo "edgeforge binary not found at $EF (set EF_BIN)" >&2; exit 2; }
[[ -f "$EFQ" ]] || { echo "no .efq file at $EFQ (run tools/quantize_tinyllama.sh)" >&2; exit 2; }
"$PY" -c 'import numpy' 2> /dev/null || { echo "$PY has no numpy: set PYTHON to the llama.cpp venv's python" >&2; exit 2; }
mkdir -p "$OUT"
cd "$OUT"
echo "== output folder: $OUT"

# ---- versions and machine
{
  echo "date        : $(date -Iseconds)"
  echo "cpu         : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
  echo "threads     : $THREADS"
  echo "llama.cpp   : $(git -C "$LLAMA" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "edgeforge   : $("$EF" --version) ($(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown))"
} | tee versions.txt

# ---- 1. GGUF models from the same BF16 checkpoint
if [[ ! -f tinyllama-bf16.gguf ]]; then
  echo "== convert BF16 checkpoint -> GGUF (bf16, exact)"
  "$PY" "$LLAMA/convert_hf_to_gguf.py" "$MODEL_DIR" --outtype bf16 --outfile tinyllama-bf16.gguf > convert.log 2>&1 \
    || { tail -20 convert.log; exit 1; }
  tail -2 convert.log
fi
if [[ ! -f tinyllama-q4_0.gguf ]]; then
  echo "== llama-quantize -> Q4_0 (llama.cpp's own recipe)"
  "$BIN/llama-quantize" tinyllama-bf16.gguf tinyllama-q4_0.gguf Q4_0 "$THREADS" > quantize.log 2>&1 \
    || { tail -20 quantize.log; exit 1; }
  tail -2 quantize.log
fi

# ---- 2. the evaluation text
if [[ -n "${TEXT:-}" ]]; then
  TEXT=$(cd "$(dirname "$TEXT")" && pwd)/$(basename "$TEXT")
else
  if [[ ! -f wikitext-2-raw/wiki.test.raw ]]; then
    echo "== download wikitext-2 (llama.cpp scripts/get-wikitext-2.sh)"
    bash "$LLAMA/scripts/get-wikitext-2.sh" > /dev/null
  fi
  TEXT=$OUT/wikitext-2-raw/wiki.test.raw
fi
echo "text        : $TEXT ($CHUNKS chunks x $CTX tokens)" | tee -a versions.txt

# ---- 3. decode speed (and peak RSS where /usr/bin/time exists)
TIMEV=()
[[ -x /usr/bin/time ]] && TIMEV=(/usr/bin/time -v)
REP_THREADS=$THREADS
for ((i = 1; i < REPS; ++i)); do REP_THREADS+=",$THREADS"; done
PROMPT=$ROOT/tests/golden/prompts/p0_factual  # the 23-token bench prompt of bench-decode and the README
[[ -f "$PROMPT/manifest.json" ]] || { echo "missing $PROMPT (golden data, README step 6)" >&2; exit 2; }
echo "== edgeforge bench-decode: golden prompt p0_factual, then 32 decode steps, $THREADS threads, $REPS runs"
${TIMEV[@]+"${TIMEV[@]}"} "$EF" bench-decode --qmodel "$EFQ" --ids-from "$PROMPT" --no-ladder --threads "$REP_THREADS" \
  --tokens 32 > ef-bench-q.log 2>&1 || { tail -20 ef-bench-q.log; exit 1; }
${TIMEV[@]+"${TIMEV[@]}"} "$EF" bench-decode --model-dir "$MODEL_DIR" --ids-from "$PROMPT" --no-ladder \
  --threads "$REP_THREADS" --tokens 32 > ef-bench-bf16.log 2>&1 || { tail -20 ef-bench-bf16.log; exit 1; }
grep -h -E '^  (avx2|scalar) ' ef-bench-q.log ef-bench-bf16.log
DEPTH=$(sed -n 's/^  prompt \([0-9]*\) tokens.*/\1/p' ef-bench-q.log)
for m in bf16 q4_0; do
  echo "== llama-bench $m: tg32 after a $DEPTH-token context (-d $DEPTH), $THREADS threads, $REPS runs"
  ${TIMEV[@]+"${TIMEV[@]}"} "$BIN/llama-bench" -m "tinyllama-$m.gguf" -t "$THREADS" -p 0 -n 32 -d "$DEPTH" -r "$REPS" \
    > "llama-bench-$m.log" 2>&1 || { tail -20 "llama-bench-$m.log"; exit 1; }
  grep -E '^\| (model|llama)' "llama-bench-$m.log"
done

# ---- 4. quality on the same text and chunks
echo "== llama-perplexity: BF16 (saves its logits), then Q4_0 vs those logits (KL divergence)"
"$BIN/llama-perplexity" -m tinyllama-bf16.gguf -f "$TEXT" -c "$CTX" --chunks "$CHUNKS" -t "$THREADS" \
  --kl-divergence-base bf16-logits.kld > ppl-bf16.log 2>&1 || { tail -20 ppl-bf16.log; exit 1; }
"$BIN/llama-perplexity" -m tinyllama-q4_0.gguf -c "$CTX" -t "$THREADS" --kl-divergence-base bf16-logits.kld \
  --kl-divergence > ppl-q4_0.log 2>&1 || { tail -20 ppl-q4_0.log; exit 1; }
grep -h -E 'Final estimate|Mean PPL|Mean +KLD|Same top' ppl-bf16.log ppl-q4_0.log || true

echo "== same input? llama.cpp's evaluated token ids (stored in bf16-logits.kld) vs EdgeForge's tokenizer"
"$EF" tokenize --tokenizer-dir "$MODEL_DIR" --text-file "$TEXT" --no-special --ids-out ef-ids.txt > ef-tokenize.log
"$PY" - bf16-logits.kld ef-ids.txt <<'PYEOF' | tee token-check.txt
import struct, sys
with open(sys.argv[1], "rb") as f:
    if f.read(8) != b"_logits_":
        sys.exit("unexpected .kld header")
    n_ctx, n_vocab, n_chunk = struct.unpack("<iii", f.read(12))
    theirs = struct.unpack("<%di" % (n_ctx * n_chunk), f.read(4 * n_ctx * n_chunk))
ours = [int(line) for line in open(sys.argv[2])][: n_ctx * n_chunk]
same = sum(a == b for a, b in zip(theirs, ours))
diff = next((i for i, (a, b) in enumerate(zip(theirs, ours)) if a != b), None)
print("token ids, %d chunks x %d: identical %d/%d%s" % (n_chunk, n_ctx, same, len(theirs),
      "" if diff is None and len(ours) == len(theirs) else "; first difference at token %s" % diff))
PYEOF

echo "== edgeforge eval-quant: same text and chunks, BF16 engine vs quantized engine"
"$EF" eval-quant --qmodel "$EFQ" --model-dir "$MODEL_DIR" --text "$TEXT" --window "$CTX" --chunks "$CHUNKS" \
  --threads "$THREADS" > ef-eval.log 2>&1 || true
grep -E 'text:|engine *: PPL|perplexity:|per position|RESULT' ef-eval.log

# ---- 5. summary.md
ls -l "$MODEL_DIR/model.safetensors" "$EFQ" > ef-files.txt
"$PY" - "$LLAMA" "$THREADS" "$REPS" "$CHUNKS" "$CTX" "$(basename "$TEXT")" "$DEPTH" <<'PYEOF' > summary.md
import re, statistics, sys
from pathlib import Path
llama, threads, reps, chunks, ctx, text, depth = sys.argv[1:8]
sys.path.insert(0, str(Path(llama) / "gguf-py"))
from gguf import GGUFReader

MiB = 1024.0 ** 2
def read(p):
    return Path(p).read_text(errors="replace") if Path(p).exists() else ""
def rss(log):
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", read(log))
    if m:
        return "%.0f MiB" % (int(m.group(1)) / 1024)
    m = re.search(r"^  peak RSS (.+)$", read(log), re.M)  # EdgeForge's own measurement
    return m.group(1) if m else "-"
def human_to_bytes(s):
    v, u = s.split()
    return float(v) * {"B": 1, "KiB": 1024, "MiB": MiB, "GiB": 1024 ** 3}[u]
def mean_sd(xs):
    return "%.2f ± %.2f" % (statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0)

def gguf_info(path):
    r = GGUFReader(path)
    total, per_token, types = 0, 0, {}
    for t in r.tensors:
        n = int(t.n_bytes)
        total += n
        types[t.tensor_type.name] = types.get(t.tensor_type.name, 0) + 1
        if t.name == "token_embd.weight":
            n //= int(t.shape[1])  # one embedding row per token
        per_token += n
    out = next((t.tensor_type.name for t in r.tensors if t.name == "output.weight"), "?")
    return Path(path).stat().st_size, per_token, types, out

def llama_bench(log):
    for line in read(log).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) > 3 and cells[-2].startswith("tg"):
            return cells[-1], float(cells[-1].split("±")[0])
    return "-", 0.0

def ef_bench(log):
    t = read(log)
    m = re.search(r"\| ([\d.]+ [KMG]?i?B) of weights per token", t)
    tps = [float(l.split()[3]) for l in t.splitlines() if re.match(r"^  (avx2|scalar) +\d+ ", l)]
    desc = t.splitlines()[0].split(": ", 1)[1] if t else "?"
    return (human_to_bytes(m.group(1)) if m else 0.0), tps, desc

print("# EdgeForge vs llama.cpp: TinyLlama-1.1B-Chat-v1.0 on one laptop\n")
print("```\n" + read("versions.txt").rstrip() + "\n```\n")
print("## Decode speed\n")
print("%s threads; 32 decode steps after a %s-token context (llama-bench `tg32 @ d%s`, EdgeForge "
      "`bench-decode`); mean ± standard deviation of %s runs. Weight GB/s = weights read per token × tok/s.\n"
      % (threads, depth, depth, reps))
print("| engine | weights | file | read / token | decode tok/s | weight GB/s | peak RSS |")
print("|---|---|---|---|---|---|---|")
for m, label in (("bf16", "BF16 GGUF"), ("q4_0", "llama-quantize Q4_0")):
    size, per_tok, types, out = gguf_info("tinyllama-%s.gguf" % m)
    shown, tps = llama_bench("llama-bench-%s.log" % m)
    kinds = ", ".join("%d %s" % (n, k) for k, n in sorted(types.items(), key=lambda kv: -kv[1]))
    print("| llama.cpp | %s (%s tensors; output %s) | %.1f MiB | %.1f MiB | %s | %.2f | %s |"
          % (label, kinds, out, size / MiB, per_tok / MiB, shown, per_tok * tps / 1e9, rss("llama-bench-%s.log" % m)))
files = {}
for line in read("ef-files.txt").splitlines():
    f = line.split()
    files[Path(" ".join(f[8:])).suffix] = int(f[4])
for log, ext in (("ef-bench-bf16.log", ".safetensors"), ("ef-bench-q.log", ".efq")):
    per_tok, tps, desc = ef_bench(log)
    gbps = per_tok * statistics.mean(tps) / 1e9 if tps else 0.0
    print("| EdgeForge | %s | %.1f MiB | %.1f MiB | %s | %.2f | %s |"
          % (desc, files.get(ext, 0) / MiB, per_tok / MiB, mean_sd(tps) if tps else "-", gbps, rss(log)))
print("\nPrompt processing is not compared: llama.cpp batches the prompt (matrix-matrix), EdgeForge "
      "processes it one token at a time (matrix-vector).\n")

print("## Quality: %s chunks × %s tokens of %s, llama-perplexity scheme\n" % (chunks, ctx, text))
print("Each engine's quantized model is compared with its own BF16 model on the same scored positions "
      "(second half of each chunk).\n")
print("```\n" + read("token-check.txt").strip() + "\n```\n")
pb = re.search(r"Final estimate: PPL = ([\d.]+)", read("ppl-bf16.log"))
q = read("ppl-q4_0.log")
pq = re.search(r"Mean PPL\(Q\)\s*: +([\d.]+)", q)
kl = re.search(r"Mean +KLD: +([\d.]+)", q)
top = re.search(r"Same top p: +([\d.]+)", q)
e = read("ef-eval.log")
eb = re.search(r"BF16 engine +: PPL ([\d.]+)", e)
eq = re.search(r"quantized engine: PPL ([\d.]+)", e)
pos = re.search(r"per position vs BF16: top-1 (\d+)/(\d+).*?KL mean ([\d.]+)", e)
res = re.search(r"RESULT: (\w+) +\((.*)\)", e)
def g(m, fmt="%s", i=1):
    return fmt % float(m.group(i)) if m and fmt != "%s" else (m.group(i) if m else "?")
def ratio(a, b):
    return "%.4f" % (float(b.group(1)) / float(a.group(1))) if a and b else "?"
print("| engine | BF16 PPL | quantized PPL | PPL ratio | mean KL vs own BF16 | top-1 same as own BF16 |")
print("|---|---|---|---|---|---|")
print("| llama.cpp (Q4_0) | %s | %s | %s | %s | %s%% |"
      % (g(pb, "%.3f"), g(pq, "%.3f"), ratio(pb, pq), g(kl, "%.4f"), g(top, "%.1f")))
print("| EdgeForge (.efq) | %s | %s | %s | %s | %s |" % (
    g(eb, "%.3f"), g(eq, "%.3f"), ratio(eb, eq), g(pos, "%.4f", 3),
    "%.1f%% (%s/%s)" % (100.0 * int(pos.group(1)) / int(pos.group(2)), pos.group(1), pos.group(2)) if pos else "?"))
if pb and eb:
    print("\nBF16 cross-check (two independent engines, same token ids and positions): perplexity %.3f "
          "(llama.cpp) vs %.3f (EdgeForge), %+.3f%%." % (float(pb.group(1)), float(eb.group(1)),
                                                     100.0 * (float(eb.group(1)) / float(pb.group(1)) - 1.0)))
print("\nEdgeForge eval-quant verdict on this text (Stage 6 targets): %s (%s)\n" % (g(res), g(res, "%s", 2)))
print("Raw logs: llama-bench-*.log, ef-bench-*.log, ppl-*.log, ef-eval.log, token-check.txt.")
PYEOF
echo
cat summary.md
echo
echo "== wrote $OUT/summary.md"
