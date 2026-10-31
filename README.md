# EdgeForge

**Bandwidth-aware 4-bit CPU inference engine for TinyLlama-1.1B.**
BIOS V2 · FRONTIER · "Your Own Inference Engine". C++20, no ML libraries in the runtime.

> Current milestone: **Stage 1: foundation** (model loader, config, memory-bandwidth
> benchmark, golden-reference tooling). No transformer maths yet.

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
tests/             self-contained test framework + suites (json, fp16, safetensors, config, membench)
tools/
  dump_reference.py            dev-only: PyTorch/transformers golden data (never linked into the engine)
  make_synthetic_checkpoint.py fake TinyLlama-shaped checkpoint for offline pipeline tests
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

## Next: Stage 2

Scalar FP32 kernels (GEMV, RMSNorm, softmax, RoPE, SwiGLU) with unit tests, then the FP32 forward pass
verified layer by layer against `tests/golden/prompts/p0_factual/`.
