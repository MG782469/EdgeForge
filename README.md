# EdgeForge

**Bandwidth-aware 4-bit CPU inference engine for TinyLlama-1.1B.**
BIOS V2 · FRONTIER · "Your Own Inference Engine". C++20, no ML libraries in the runtime.

> Current milestone: **Stage 6: the engine running straight from the packed 4-bit `.efq` weights.**
> Checkpoints: `stage1-verified` (loader, membench, golden data), `stage2-verified` (math kernels),
> `stage3-verified` (full forward pass + KV cache), `stage4-verified` (tokenizer + chat),
> `stage5-verified` (mixed-precision quantizer: 12/12 checks, 154/154 matrices, 713.0 MiB).

## Stage 6: quantized inference (`include/ef/qkernels.h`, `model.h`, `eval_quant.h`)

```bash
D=~/edgeforge-data; M=$D/tinyllama; Q=$D/tinyllama-q4.efq      # Q from tools/quantize_tinyllama.sh
~/ef-build/edgeforge eval-quant --qmodel $Q --model-dir $M --golden tests/golden --text tests/fixtures/eval_text.txt
~/ef-build/edgeforge generate --qmodel $Q --ids-from tests/golden/prompts/p0_factual --max-new 16   # 4-bit engine
~/ef-build/edgeforge generate --model-dir $M --ids-from tests/golden/prompts/p0_factual --max-new 16 # BF16 engine
~/ef-build/edgeforge chat --qmodel $Q --tokenizer-dir $M                                            # 4-bit chat
```

**How it runs.**
- `Model::load_efq` maps the `.efq` file and points every matrix at its packed bytes (`WeightView`:
  type, data, rows, cols). **Nothing is dequantized ahead of time**: no FP32 copy of the model exists.
- `kern::matvec` picks the kernel per tensor, because the file is mixed precision (148 Q4_0 + 6 Q8_0 layer
  matrices, Q8_0 lm_head, BF16 embedding).
  - **BF16:** exactly the Stage 2 kernel. The reference engine is unchanged: `verify-forward` output is
    identical before and after this stage.
- Packed dot products read the 18-byte Q4_0 / 34-byte Q8_0 blocks in place:
  `Q4_0: d · Σ (code_j − 8) · x_j`, `Q8_0: d · Σ q_j · x_j`.
  - The scale is applied once per block.
  - 32 FP32 lane accumulators run across the row, with a fixed final summation order.
  - The scalar C++ is shaped so GCC/Clang auto-vectorise it.
  - **Per-weight speed:** equal to the BF16 kernel. Sandbox micro-benchmark, 5632×2048: BF16 3.05,
    Q4_0 2.89, Q8_0 2.77 Gweights/s. Each Q4_0 weight costs 0.56 bytes instead of 2.
- Chat, the KV cache and multi-turn reuse work unchanged on the quantized model: `ChatSession` only sees
  a `Model`.

**`eval-quant`: what is measured.**
1. **Kernels on the real weights:** 16 sampled rows of every Q4_0/Q8_0 matrix. The packed dot must equal
   the dequantize-then-dot in double precision (relative error ≤ 1e-5 of Σ|w·x|).
2. **vs PyTorch FP32 (golden), teacher-forced.**
   - Positions: every position of `p0`, the last position of `p1`–`p3`, and 2 decode steps each.
   - Metrics: KL(reference ‖ quantized), top-1 agreement, and the greedy continuation of every prompt.
     For greedy, the KV cache is rewound at the first difference and our own continuation is shown.
3. **Text perplexity** on `tests/fixtures/eval_text.txt` (first 128 tokens), BF16 engine vs quantized
   engine, with KL and top-1 at every position.
4. **Speed and memory:** tok/s and GB/s of weights for both engines, and the quantized engine's peak RSS.

**Quality targets, fixed before the first real-model run:**

| Target | Bound | Why |
|---|---|---|
| Mean KL(reference ‖ quantized) | ≤ 0.10 nats per position | the distribution must stay close everywhere |
| Top-1 agreement where the reference is confident (p ≥ 0.5) | ≥ 95% | when the model is sure, quantization must not change the answer |
| Text perplexity ratio | ≤ 1.15 | at most 15% more surprised by ordinary text |

Plain top-1 agreement and greedy identity are **reported, not targets**: at near-ties any noise flips the
argmax, and one flip changes the whole continuation.

**Tests (all on tiny models, no real weights needed):**
- packed kernels vs dequantized weights on random blocks of every bit pattern, the nibble layout by hand,
  and dispatch for all 5 types;
- a lossless `.efq` (all BF16) runs **bit-identically** to the safetensors engine (tied and untied);
- the quantized engine matches an F32 copy of its own dequantized weights (max rel err 1.7e-6, same greedy
  tokens);
- `eval-quant` on a lossless file gives KL 0, PPL ratio 1, and identical greedy tokens.

**Pipeline check on a full-size synthetic checkpoint** (random weights, so the quality numbers mean
nothing; 1 sandbox core):
- quantized engine 2.34 tok/s vs BF16 2.02 tok/s (1.16×);
- quantized engine peak RSS 594–620 MiB vs a 2.2 GB BF16 model.

Scalar single-thread decode is **compute-bound** (BF16 reaches only ~4 GB/s here). So 3.4× fewer bytes
gives just 1.16× more tok/s. Once Stage 7's SIMD and threads make decoding memory-bound, the byte saving
turns into speed.

Real-weight numbers come from the WSL run.


## Stage 5: quantization (`include/ef/quant.h`, `qfile.h`, `quantizer.h`)

**Reproduce the verified model** from the original BF16 checkpoint: one command, no manual steps. It
quantizes with the named policy, runs the full verification, and exits non-zero unless it passes.

```bash
bash tools/quantize_tinyllama.sh ~/edgeforge-data/tinyllama ~/edgeforge-data/tinyllama-q4.efq
# equivalent to:
~/ef-build/edgeforge quantize --model-dir ~/edgeforge-data/tinyllama --out ~/edgeforge-data/tinyllama-q4.efq \
    --policy tinyllama-1.1b-chat-v1.0
~/ef-build/edgeforge verify-quant --qmodel ~/edgeforge-data/tinyllama-q4.efq \
    --model-dir ~/edgeforge-data/tinyllama --golden tests/golden
```

The output is byte-identical for any thread count, so re-running gives the same sha256 (the script
prints it). The verified file (WSL laptop, `stage5-verified` binary) has sha256
`83c742061eea567a7387baf4a676e70328c021abe9f78715cc2a09948af08e80`.

**Verified on the real TinyLlama-1.1B-Chat-v1.0 weights (WSL laptop, 2026-10-31).**

| Metric | Result |
|---|---|
| verify-quant | **12/12 PASS** |
| Layer matrices on real activations | **154/154 PASS** (cosine ≥ 0.99 on every prompt) |
| Size | 2.05 GiB → **713.0 MiB** (2.94× smaller) |
| Weights streamed per token | 1.93 GiB → **588.0 MiB** (3.36× less) |
| Matrices in Q4_0 / Q8_0 | 148 / 6 (+ Q8_0 lm_head) |
| Worst Q4_0 weight rel RMSE | 0.1171 (bound 0.12) |
| Worst Q8_0 weight rel RMSE | 0.0090 (bound 0.01) |
| Worst matrix cosine | 0.990002: layer 19 `v_proj` (bound 0.99; its worst single position is 0.908686) |

**The margins are thin, and these numbers belong to this checkpoint.**
- All three bounds pass with little headroom.
- **What the 0.99 means:** it is an aggregate, matrix-level cosine. For each matrix and each prompt, the
  outputs of all of that prompt's positions are compared together. It is **not** a per-position guarantee
  (layer 19 `v_proj` reaches 0.908686 at one position), and **not** an end-to-end one: the errors of 154
  matrices compound through 22 layers.
- End-to-end quality (top-1 agreement, KL, perplexity vs the BF16 engine) is measured in Stage 6.
- The 3.36× is the reduction in bytes read per token, an upper bound on the decode speed-up. Real speed is
  measured in Stages 6–7.

**Precision policy.**

| Part | Stored as | Bits per weight | Why |
|---|---|---|---|
| 148 layer matrices | `Q4_0` | 4.5 | the bulk of the bytes read per token |
| layer 0 `k_proj`; `v_proj` of layers 0, 1, 2, 17, 18 | `Q8_0` (overrides) | 8.5 | each failed a bound under all-Q4_0 (below); +1.5 MiB total |
| lm_head | `Q8_0` | 8.5 | protects the logits |
| Embedding | BF16 (lossless) | 16 | only one row is read per token |
| Norms | F32 | 32 | |

**Named policies.**
- They are version-controlled in `named_policies()` (`src/quantizer.cpp`) and recorded in the `.efq`
  header.
- `--policy` values:
  - `tinyllama-1.1b-chat-v1.0`: the 6 overrides above. It has an architecture guard (2048 / 5632 / 22
    layers / 32 heads / 4 KV heads / 32000 vocab, untied) and is refused on any other config.
  - `uniform`: no per-tensor exceptions, works for any model.
  - `auto` (default): picks a named policy only when its architecture guard matches, and prints that;
    otherwise `uniform`. The library itself (`QuantizeOptions`) never adds model-specific exceptions.
- `--override TENSOR=TYPE,...` adds exceptions on top of the policy (exact names or `*` globs, last match
  wins). The header then records `<policy>+custom`.
- The overrides were measured on this checkpoint. Another checkpoint with the same architecture gets the
  same 6 Q8_0 matrices (they cost 1.5 MiB and never lower precision), but needs its own `verify-quant` run.

**How the policy was found (thresholds never changed).**
1. **All-Q4_0, layer-0 check only: 8/10.**
   - Layer-0 `k_proj` weight rel RMSE was **0.1256** > 0.12.
   - Layer-0 `v_proj` GEMV cosine was **0.978525** < 0.99.
2. **Layer-0 `k_proj`/`v_proj` in Q8_0, check extended to all 154 matrices:** 4 more `v_proj` failed
   (layers 1, 2, 17, 18).
3. **Those promoted to Q8_0, as measured and suggested by `verify-quant`:** 12/12 and 154/154.

Notes:
- **MSE scale search vs absmax:** lower *weight* error (sampled rel RMSE 0.0914 vs 0.0948), but a slightly
  lower worst layer-0 *activation* cosine (0.978525 vs 0.978716). Minimising weight error is not
  minimising output error: inputs with outlier channels amplify the error in those columns.
- **Why `v_proj` gets the extra bits:** GQA makes k/v small (256 × 2048 per layer), so each costs only
  256 KiB more in Q8_0.

**Q4_0 format.**
- **Block:** an FP16 scale plus 16 bytes of nibbles. `w[j] = (lo(qs[j]) - 8)·d` and
  `w[j+16] = (hi(qs[j]) - 8)·d`, so unpacking is one AND and one shift (AVX2-friendly).
- **Scale per block:** we try llama.cpp's `max/-8` plus 5 symmetric candidates, then refine by least
  squares (codes fixed, `d = Σx·c / Σc²`). Including `max/-8` means the weight error is never worse than
  plain Q4_0.
- **FP16 rounding:** the scale is rounded to FP16 *before* the codes are computed, so the stored pair is
  exactly what dequantization uses.
- **Options:** `--no-search` keeps plain absmax. `quant-analyze --model-dir M --golden tests/golden`
  compares g32/g64/g128 and absmax vs MSE.

**`.efq` file.**
- Layout: magic `EFQ1`, version, a JSON header, then tensors.
- The JSON header holds the quant policy (defaults, policy name, `overrides`), the verbatim `config.json`, and a tensor
  table with type, **rule** (`layers`, `lm_head`, `embed`, `norm`, `override`, `fallback-bf16`), shape,
  offset, bytes and the measured error.
- Every tensor is 64-byte aligned, in forward-pass order, so decoding streams the file front to back.
- Loading is mmap-based, with full validation: magic, version, bounds, alignment, sizes per type,
  overlap, the expected tensor set and shapes, and **every tensor's type re-derived from the policy**
  (an exception is never silent; a mismatch is rejected).
- `.efq` files are git-ignored.

**`verify-quant` checks:**
- every tensor is dequantized and compared with the BF16 source:
  - Q4 rel RMSE must be ≤ 0.12 and Q8 ≤ 0.01 (the 5 largest Q4 errors are listed);
  - lossless tensors must be bit-identical;
  - no NaN/Inf;
  - the error stored in the header must match;
- re-quantizing sample rows reproduces the file byte for byte (the output is also identical for any thread
  count);
- size and weight bytes per token are computed from the actual tensor types, plus the cost of the
  overrides;
- with `--golden`:
  - layer-0 GEMV outputs vs the **PyTorch** layer-0 outputs (cosine ≥ 0.99);
  - **all 154 layer matrices** on real activations, re-derived from the golden residual stream with the
    exact BF16 weights. `p0` (all 23 positions) feeds all 7 projections per layer, and `p1`–`p3` (last
    position) feed q/k/v. Recomputing each layer must reproduce the golden layer output (≤ 1e-3),
    otherwise no output error is reported. Every matrix needs cosine ≥ 0.99 on every prompt
    (aggregated over that prompt's positions; the worst single position is printed for information);
  - every override vs the uniform policy: weight error, output cosine, and bytes;
  - lm_head top-1 on all 4 prompts;
- for any failure, the next precision is measured and a complete, ready command
  (`--policy uniform --override ...`) is printed. Nothing is applied automatically.

**Pipeline check on a full-size synthetic checkpoint (random BF16 weights, 2 cores):**
- Quantize takes 27 s with a warm page cache; two runs give byte-identical files. verify-quant takes 15 s,
  10 s of it for the activation checks.
- Method comparison (rel RMSE on random weights):

  | Method | Rel RMSE |
  |---|---|
  | g32 absmax | 0.0859 |
  | g32 MSE (ours) | 0.0824 |
  | g64 | 0.0922 |
  | g128 | 0.1007 |


## Stage 4: tokenizer and chat (`include/ef/tokenizer.h`, `sampler.h`, `chat.h`)

```bash
M=~/edgeforge-data/tinyllama
~/ef-build/edgeforge chat --model-dir $M                                   # interactive, greedy
~/ef-build/edgeforge chat --model-dir $M --prompt "What is the capital of France?"   # one-shot
~/ef-build/edgeforge chat --model-dir $M --temp 0.7 --top-p 0.9 --seed 1   # sampling
~/ef-build/edgeforge tokenize --tokenizer-dir $M --text "Hello नमस्ते 😀"
~/ef-build/edgeforge verify-tokenizer --tokenizer-dir tests/fixtures/tinyllama-tokenizer --golden tests/golden \
                                      --model-dir $M                        # + end-to-end text vs PyTorch
```

- **Tokenizer:**
  - reads `tokenizer.json`: 32,000 pieces and 61,249 merges;
  - BPE in merge-rank order, byte fallback to `<0xXX>`, special-token splitting;
  - `legacy=false` behaviour: `▁` is prepended only at the start of the text, never after `</s>`.
- **Decoder:**
  - `▁` becomes a space, byte runs become UTF-8 (invalid runs become U+FFFD per byte, like HF);
  - one leading space is stripped;
  - the streaming decoder holds back a byte run until it is complete, so streamed text always equals the
    final decode.
- **Chat:**
  - TinyLlama/Zephyr template, byte-identical to the Jinja one;
  - multi-turn, re-uses the KV cache through the longest common token prefix, so earlier turns are not
    recomputed;
  - drops the oldest turns when the context is full;
  - commands `/reset` and `/exit`.
- **Sampler:**
  - greedy by default;
  - `--temp`, `--top-k`, `--top-p`, `--rep-penalty`, `--seed`, applied in HF order.
- **Verified against HF:**
  - 122/122 strings for ids, pieces, decode and decode skipping specials;
  - 10/10 chat-template cases;
  - 4/4 golden prompts, text → ids and ids → text.
  - The TinyLlama tokenizer files are committed in `tests/fixtures/` (Apache-2.0), so CTest runs this
    without the model.
- **Speed:** prompt processing is still one token at a time (scalar, 1 thread): about 9 s for a 23-token
  prompt. Batched prefill, threads and SIMD come in Stage 7.


## Stage 3: forward pass (`include/ef/model.h`)

- **`Model`**:
  - a zero-copy view of the memory-mapped BF16 weights;
  - every tensor is validated against config.json;
  - computes the bytes streamed per token (1.93 GiB for TinyLlama).
- **`Transformer::forward(token, pos)`**:
  - one token through all 22 layers, made only from the Stage 2 kernels;
  - KV cache layout `[layer][pos][256]`;
  - writing at `pos < cached()` rewinds the cache;
  - lm_head is skipped for prefill positions whose logits are not needed.
- **`generate_greedy`**: prefill, then argmax decoding until EOS or `max_new`.

```bash
M=~/edgeforge-data/tinyllama
# all prompts: hidden state of every layer, logits at every position, decode logits through the
# KV cache, greedy tokens (scalar 1 thread: about a minute)
~/ef-build/edgeforge verify-forward --golden tests/golden --model-dir $M
# first FP32 baseline speed (no tokenizer yet: prompt given as ids)
~/ef-build/edgeforge generate --model-dir $M --ids-from tests/golden/prompts/p0_factual --max-new 16 --bench bench/membench.json
```

Tolerances against the FP32 PyTorch reference:
- **hidden states:** rel ≤ 1e-3 and cosine ≥ 0.9999;
- **logits:** rel ≤ 1e-3 and cosine ≥ 0.99999;
- **argmax:** identical at every position;
- **greedy tokens:** identical. A flip is tolerated only where HF's own top-1/top-2 margin is below 1e-3.

How the forward pass was tested:
- `ef_tests model` compares against a double-precision reference written separately in the test. That
  reference recomputes the whole sequence **without** a KV cache, so it independently checks positions,
  caching, GQA and tied embeddings.
- An injected KV-offset bug makes four of those tests fail.

Measured on a synthetic 2.2 GB checkpoint: 2.6 tok/s scalar, about 5.3 GB/s of weights.



## Stage 2: math kernels (`include/ef/kernels.h`)

| Kernel | HF behaviour matched |
|---|---|
| `matvec_f32` / `matvec_bf16` | y = W·x, W row-major [out][in]. FP32 accumulation with 8 partial sums. BF16 is widened exactly, so it is bit-identical to the FP32 path. |
| `rmsnorm` | `w * (x * rsqrt(mean(x²) + eps))` |
| `softmax_inplace` | Max-subtracted; masked `-inf` entries become exactly 0 |
| `RopeTable` + `rope_apply` | FP32 `inv_freq` and angles as in HF. Half-split pairing (i, i+32). |
| `silu` / `swiglu` | `silu(gate) * up`. Overflow-safe for very negative inputs. |
| `attention` | Causal, GQA (head h → KV head h/8), scale 1/√64, one query against a KV cache |
| `embedding_bf16`, `add_inplace`, `argmax` | Exact lookup, residual add, first-max argmax |

Two levels of checks:

- **Unit tests:** `ef_tests kernels` covers known values and float64 references. Error is checked against a
  rigorous FP32 bound. The tests also cover masking, overflow and odd sizes, plus RoPE properties:
  identity at pos 0, norm preservation, scores that depend only on relative position.
- **`edgeforge verify-math`:** compares each kernel with the PyTorch golden tensors from Stage 1.

```bash
# weight-free: RoPE, softmax+GQA attention, SwiGLU, residual (needs only tests/golden)
~/ef-build/edgeforge verify-math --golden tests/golden
# full: + embedding, RMSNorm x3, all 7 layer-0 GEMVs, a CHAINED layer 0 from token ids, final norm, lm_head
~/ef-build/edgeforge verify-math --golden tests/golden --model-dir ~/edgeforge-data/tinyllama --bench bench/membench.json
```

Tolerances against the FP32 reference:
- elementwise ops: max error ≤ 1e-5 of the tensor's max |x|;
- GEMV and attention: ≤ 1e-4, with cosine ≥ 0.99999;
- embedding: exact;
- lm_head: the top-1 and top-5 tokens must be identical.

CTest runs `golden_math` automatically when `tests/golden` exists. Configure with
`-DEF_MODEL_DIR=...` to include the weight checks. Without the real model, `tools/make_synthetic_golden.py`
builds float64 numpy golden data for a synthetic checkpoint, which exercises the weight path.

## Stage 1 status

| Component | Status | Where it was verified |
|---|---|---|
| CMake + C++20 project, `-Werror`-clean | done | GCC 13, Clang 18, GCC 11 (laptop sandbox) |
| Strict JSON parser (dup keys, UTF-8, surrogates, depth) | done | 10 tests + 20k-case fuzz |
| Safetensors reader (mmap, full validation, unaligned-safe) | done | 11 tests + 30k-case fuzz, ASan/UBSan clean |
| BF16/FP16 to F32 conversions | done | all 65,536 bit patterns + 2M compiler cross-checks |
| config.json parser + expected-tensor table | done | TinyLlama acceptance numbers in tests |
| `inspect` / `verify` CLI | done | **full-size synthetic checkpoint** with the exact TinyLlama layout (2.2 GB) |
| Multithreaded membench (checksum-verified) | done | workspace sanity run; **laptop run pending** |
| `tools/dump_reference.py` (golden data) | written | numpy self-test passes; **torch run pending (needs your laptop)** |
| Real TinyLlama checkpoint | **pending** | Hugging Face is not reachable from the build sandbox: run step 4 |

## Layout

```
include/ef/        public headers (json, fp16, mapped_file, safetensors, model_config, membench, sysinfo)
src/               implementation + main.cpp (CLI)
tests/             self-contained test framework + suites (json, fp16, safetensors, config, membench,
                   kernels, golden); tests/golden = PyTorch reference data from Stage 1
tools/
  dump_reference.py            dev-only: PyTorch/transformers golden data (never linked into the engine)
  make_synthetic_checkpoint.py fake TinyLlama-shaped checkpoint for offline pipeline tests
  make_synthetic_golden.py     float64 numpy layer-0 golden data for a synthetic checkpoint
  build.sh                     build + test without CMake
  quantize_tinyllama.sh        reproduce + verify the Stage 5 model from the BF16 checkpoint
tests/fixtures/eval_text.txt   fixed text for the Stage 6 perplexity comparison
docs/              architecture.md, golden_format.md
```

## 1. Set up (Windows laptop: WSL Ubuntu recommended)

PowerShell (admin), once: `wsl --install -d Ubuntu`. Then inside Ubuntu:

```bash
sudo apt update && sudo apt install -y build-essential cmake ninja-build python3-pip python3-venv
```

**Keep the model and build folders out of OneDrive.** The code can stay in
`C:\Users\BIT\OneDrive\Desktop\bios Hackathon` (WSL path below). Put the 2.2 GB model in WSL's own disk
(`~/edgeforge-data`). Reading model files through `/mnt/c` is slow, and OneDrive may
upload or evict them.

```bash
SRC="/mnt/c/Users/BIT/OneDrive/Desktop/bios Hackathon"
```

Native Windows alternative (Visual Studio 2022, "Desktop development with C++"): `cmake -S . -B C:\ef-build`,
`cmake --build C:\ef-build --config Release`, `ctest --test-dir C:\ef-build -C Release`. This MSVC
path has **not been tested yet**.

## 2. Build and test

```bash
cmake -S "$SRC" -B ~/ef-build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build ~/ef-build
ctest --test-dir ~/ef-build --output-on-failure
```

No CMake? `bash "$SRC/tools/build.sh" ~/ef-build` builds both binaries and runs the tests.
Options: `-DEF_WERROR=ON`, `-DEF_SANITIZE=ON` (ASan + UBSan), `-DEF_NATIVE=OFF`.

## 3. Download TinyLlama

```bash
python3 -m pip install -U huggingface_hub
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ~/edgeforge-data/tinyllama
# older huggingface_hub: huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ~/edgeforge-data/tinyllama
sha256sum ~/edgeforge-data/tinyllama/model.safetensors   # compare with the SHA256 on the HF file page
```

## 4. Inspect and verify the real checkpoint

```bash
M=~/edgeforge-data/tinyllama
~/ef-build/edgeforge inspect --model-dir $M
~/ef-build/edgeforge inspect --model-dir $M --tensor model.layers.0.self_attn.q_proj.weight
~/ef-build/edgeforge verify  --model-dir $M
```

`verify` must print `RESULT: PASS`. Hard acceptance numbers: **201 tensors, 1,100,048,384 parameters,
2,200,096,768 tensor bytes, all BF16, no NaN/Inf.** `inspect --tensor` also prints the process peak RSS.
It stays far below 2 GB because the file is memory-mapped and only the requested tensor is read.

## 5. Memory bandwidth (run on the demo laptop)

```bash
mkdir -p "$SRC/bench"
~/ef-build/edgeforge membench --json "$SRC/bench/membench_$(hostname).json"
```

Defaults: 1 GiB working set, thread sweep (1, 2, 4, physical cores, logical CPUs), 5 timed runs × 3
passes, 1 warm-up, and every run's checksum is verified. Before running, plug the laptop in, set Windows
power mode to *Best performance* and close other apps. The printed **peak median GB/s** is the
denominator for bandwidth efficiency in later stages. It is software-measured read bandwidth, so it
will be below the RAM's spec-sheet number.

## 6. Golden reference data (dev-only; needs PyTorch)

First confirm with the organisers that PyTorch may be used as an **offline test oracle**. It is never
part of the engine.

```bash
python3 -m venv ~/ef-venv && source ~/ef-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r "$SRC/requirements-dev.txt"
cd "$SRC"
python tools/dump_reference.py --selftest
python tools/dump_reference.py --check-env --model-dir ~/edgeforge-data/tinyllama
python tools/dump_reference.py --model-dir ~/edgeforge-data/tinyllama --out-dir tests/golden
~/ef-build/edgeforge verify --model-dir ~/edgeforge-data/tinyllama --golden tests/golden
```

- **FP32 reference by default.** It needs about 5.5 GB of free RAM. `--dtype bf16` is a labelled fallback only.
- **Expected output:** `attention self-check: PASS`. The script recomputes layer-0 attention with
  half-split RoPE plus GQA and must match HF. It also prints how far off the *wrong* adjacent-pair RoPE
  would be.
- **Weight probe:** `verify --golden` then cross-checks sampled weight bits (bit-exact) and full-tensor
  sums against the official `safetensors` reader.
- **Format:** see `docs/golden_format.md`.

Then add the real-model test to CTest:
`cmake -S "$SRC" -B ~/ef-build -DEF_MODEL_DIR=$HOME/edgeforge-data/tinyllama -DEF_GOLDEN_DIR="$SRC/tests/golden"`.

## Offline pipeline test without the real model

```bash
python3 tools/make_synthetic_checkpoint.py --out /tmp/fake-tinyllama      # 2.2 GB, random BF16
~/ef-build/edgeforge verify --model-dir /tmp/fake-tinyllama --golden /tmp/fake-tinyllama/golden
```

## Stage 1 acceptance checklist

- [x] Project compiles (GCC 13 / Clang 18 with `-Werror`; GCC 11 on the laptop sandbox)
- [x] Automated tests pass: 38 tests, 6,131 checks; ASan + UBSan clean; parsers fuzzed
- [x] Config parsing works and is validated (GQA, head_dim, unsupported features rejected)
- [x] BF16 conversion validated exhaustively
- [x] Loader validated end-to-end at full TinyLlama scale (synthetic weights): probe 256/256 bit-exact
- [x] Benchmark produces checksum-verified measurements (workspace sanity run)
- [x] No ML inference libraries in the C++ runtime (standard library + OS APIs only)
- [ ] **Real TinyLlama tensors inspected + `verify` PASS** (step 4, on the laptop)
- [ ] **membench on the demo laptop, native, plugged in** (step 5)
- [ ] **Golden reference generated + self-check PASS** (step 6, after the organisers' OK)

## Known limitations

- Single-file `model.safetensors` only (no sharded checkpoints); little-endian hosts only.
- The MSVC build is untested. On Windows, paths with non-ASCII characters are not supported (narrow `argv`).
- membench measures read bandwidth only, which is what decode is bound by. It is not a DRAM hardware counter.
- `dump_reference.py`'s PyTorch path could not be executed in the build sandbox (no PyTorch / Hugging Face
  access there). Its numpy maths self-test passes, and the probe format is exercised by `verify` on the
  synthetic checkpoint.

## Next: Stage 7

Speed:
- explicit AVX2 kernels for Q4_0 / Q8_0 (and optionally INT8-quantized activations);
- multi-threaded GEMV across rows;
- the speed-up ladder scalar → SIMD → threads, measured in tok/s and in GB/s against the membench
  peak.

The outputs must stay within the Stage 6 targets, checked with the same `eval-quant`.
