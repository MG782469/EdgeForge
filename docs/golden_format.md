# Golden reference format (`edgeforge-golden-v1`)

Produced by `tools/dump_reference.py` (dev-only, PyTorch + transformers, CPU, **FP32** by default).
Consumed by EdgeForge tests from Stage 2 onward.

## Binary tensors

- Raw little-endian, C order, no header.
- `*.f32` = float32, `*.i32` = int32.
- Shape, dtype and a description of every file are listed in the `manifest.json` next to it.

## Layout

```
tests/golden/
  meta.json                versions, sha256 of model.safetensors + config.json, reference dtype,
                           attention self-check result, full config
  weight_probe.json        official safetensors reader: sampled raw bits + float64 sums per tensor
  tokenizer_cases.json     122 strings: ids (no specials), ids_default (with BOS), pieces, decoded text
  chat_template_cases.json conversations -> exact template text + ids (BOS + text)
  prompts/
    p0_factual/            the detail prompt (all positions + layer-0 internals)
    p1_technical/  p2_code/  p3_hindi/
```

## Per-prompt files

| file | shape | meaning |
|---|---|---|
| `input_ids.i32` | [S] | BOS + tokenised chat template (exactly one BOS) |
| `embed_last.f32` | [D] | embedding of the last prompt token |
| `hidden_last.f32` | [L, D] | output of decoder layer *l* at the last position |
| `final_norm_last.f32` | [D] | final RMSNorm output, last position |
| `logits_last.f32` | [V] | next-token logits after the prompt |
| `decode_logits.f32` | [n, V] | logits after feeding greedy tokens 1..n through the KV cache |

Detail prompt only (`p0_factual` by default):

| file | shape | meaning |
|---|---|---|
| `embed_all`, `final_norm_all` | [S, D] | all positions |
| `hidden_all` | [L, S, D] | every layer, every position |
| `logits_all` | [S, V] | logits at every position (for prefill vs decode consistency) |
| `rope_cos`, `rope_sin` | [S, 64] | HF half-split tables: dim *i* pairs with *i + 32* |
| `layer0/input_norm` | [S, D] | input RMSNorm |
| `layer0/q`, `layer0/k`, `layer0/v` | [S, 2048], [S, 256], [S, 256] | projections **before** RoPE |
| `layer0/q_rope`, `layer0/k_rope` | same | after RoPE (validated by the self-check) |
| `layer0/attn_context` | [S, D] | attention output before `o_proj` |
| `layer0/attn_out` | [S, D] | `o_proj` output |
| `layer0/resid_after_attn` | [S, D] | x + attention |
| `layer0/post_attn_norm` | [S, D] | post-attention RMSNorm |
| `layer0/gate`, `layer0/up`, `layer0/mlp_act` | [S, 5632] | gate, up, silu(gate)·up |
| `layer0/mlp_out`, `layer0/out` | [S, D] | MLP output, layer output |

`manifest.json` also has:
- the prompt and the exact formatted text;
- the greedy result: ids, text, whether it stopped on EOS;
- for every step, the top-5 `(id, logit)` and the top1-top2 margin. A late divergence on a near-tie
  margin is acceptable; an early one is a bug.

## Suggested tolerances (FP32 engine vs FP32 reference)

- **Elementwise / norm outputs:** max abs error ≤ 1e-4 relative to the tensor's max |x|.
- **Layer outputs:** cosine similarity ≥ 0.99999 and max abs error ≤ 1e-3. Error grows slowly with depth.
- **Logits:** top-1 identical; max abs error ≤ 1e-2.
- **Greedy tokens:** identical for at least the first ~30 tokens, unless a step's margin is < 1e-3.

## Weight probe (`weight_probe.json`)

```json
{"format": "edgeforge-weight-probe-v1", "file": "model.safetensors",
 "tensors": [{"name": "...", "dtype": "BF16", "shape": [...], "numel": N,
              "samples": [[flat_index, raw_bits], ...], "first8_f32": [...],
              "sum_f64": s, "sum_abs_f64": a}]}
```

`edgeforge verify --golden DIR` requires:
- every sample to be bit-identical;
- the full-tensor sums to match within 1e-9·max(1, sum_abs).
