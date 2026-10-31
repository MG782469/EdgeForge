# EdgeForge architecture

## Pipelines

```
A. once:        model.safetensors (BF16) -> loader -> quantizer (policy tinyllama-1.1b-chat-v1.0: Q4_0 g32
                layers, 6 measured-sensitive matrices + lm_head in Q8_0, BF16 embed, F32 norms) -> model.efq
                (tools/quantize_tinyllama.sh reproduces and verifies it)
B. per prompt:  text -> tokenizer -> [embed -> 22 x (RMSNorm, QKV, RoPE, GQA attn + KV cache, o_proj,
                RMSNorm, SwiGLU MLP) -> final norm -> lm_head] -> sampler -> text  (+ tok/s, GB/s, % of peak)
```

## Stage plan

| # | Stage | Done when |
|---|---|---|
| 1 ✅ | Foundation: loader, config, membench, golden tooling | real checkpoint `verify` PASS, laptop membench, golden + self-check PASS |
| 2 ✅ | Scalar FP32 kernels | `edgeforge verify-math` PASS incl. chained layer 0 + lm_head |
| 3 ✅ | FP32 forward pass + KV cache | `edgeforge verify-forward` PASS on all golden prompts |
| 4 ✅ | Tokenizer, chat template, sampler | `verify-tokenizer` 122/122 + chat template + end-to-end text vs PyTorch |
| 5 ✅ | Quantizer + `.efq` file format | `verify-quant` 12/12 on the real weights: reconstruction, determinism, GEMV cosine of all 154 layer matrices on real activations |
| 6 | Q4 inference | top-1 agreement / KL vs FP32 engine |
| 7 | Threads + SIMD (+ INT8 activations) | same outputs, speed-up ladder |
| 8 | Profiler, benchmarks, demo | reproducible report |

## Stage 1 design decisions

- **mmap, never read-all.** `MappedFile` maps the 2.2 GB file; the OS pages in only what is touched.
  `inspect --tensor` reports peak RSS to prove it.
- **Strict validation, like the reference implementation.** The safetensors loader checks:
  - header ≤ 100 MB;
  - no duplicate names;
  - known dtypes only;
  - shape × dtype size equals the byte span;
  - offsets inside the file;
  - tensors tile the data region exactly (no overlap, gap or trailing bytes).

  A corrupt or truncated download fails loudly with the tensor name and offset.
- **Unaligned-safe.** Tensor data starts at `8 + N` and can be at any alignment, so every element read
  uses `memcpy`. The Stage 5 file format will add 64-byte alignment for SIMD.
- **Exact conversions.** BF16 to F32 is a 16-bit shift. FP16 handles subnormals. F32 to BF16/FP16
  rounds to nearest-even. All 65,536 inputs are tested.
- **Config-driven expectations.** `expected_tensors(cfg)` derives every name and shape from config.json.
  `verify` adds TinyLlama's hard numbers: 201 tensors, 1,100,048,384 params, 2,200,096,768 bytes.
- **Benchmark integrity.**
  - Each run sums a known pattern, and the checksum must equal a closed-form value, so reads cannot be
    optimised away or silently wrong.
  - A compiler barrier between passes forces real re-reads.
  - Thread start-up is outside the timed region, which is bounded by `std::barrier` phases.

## Interface contract for later stages (agree before parallel work)

```cpp
// All weight matrices row-major [out_features][in_features] (as in the checkpoint).
struct BlockQ4 { uint16_t d; uint8_t qs[16]; };  // 32 weights, 4.5 bits/weight
// w[j]    = ((qs[j] & 0xF) - 8) * fp16(d)        for j in 0..15
// w[j+16] = ((qs[j] >> 4)  - 8) * fp16(d)        (low/high nibble split: AVX2 unpack = AND + shift)
struct BlockQ8 { uint16_t d; int8_t qs[32]; };   // 8.5 bits/weight (lm_head + overridden layer matrices)
// Mixed precision: every .efq tensor carries its own type. Stage 6 must dispatch per tensor
// (Q4_0 / Q8_0 / BF16 / F32) - never assume "all layer matrices are Q4_0".

void matvec_f32(float* y, const float* W, const float* x, int rows, int cols);
void matvec_q4 (float* y, const BlockQ4* W, const float* x, int rows, int cols);
void matvec_q8 (float* y, const BlockQ8* W, const float* x, int rows, int cols);

// KV cache: k[layer][pos][n_kv_heads * head_dim], v likewise (FP32 first, FP16 later).
// RoPE: Hugging Face half-split - dim i rotates with dim i + head_dim/2 (NOT adjacent pairs).
// GQA: query head h uses KV head h / 8.
float* forward(Model& m, RunState& s, int token, int pos);  // appends this token's K/V
```
