# EdgeForge

**Bandwidth-aware 4-bit CPU inference engine for TinyLlama-1.1B.**
BIOS V2 · FRONTIER · "Your Own Inference Engine". C++20, no ML libraries in the runtime.

> Current milestone: **Stage 3: the complete FP32 forward pass with a KV cache**, verified against PyTorch.
> Checkpoints: `stage1-verified` (loader, membench, golden data), `stage2-verified` (math kernels).

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

## Next: Stage 4

- C++ tokenizer (`tokenizer.json`, SentencePiece-style BPE + byte fallback), verified against
  `tokenizer_cases.json` (122 strings) and `chat_template_cases.json`.
- TinyLlama chat template, sampler and a terminal chat.
