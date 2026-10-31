#!/usr/bin/env python3
"""EdgeForge - golden layer-0 data for a SYNTHETIC checkpoint, computed in numpy float64.

Exercises `edgeforge verify-math --model-dir ...` (the weight-dependent path)
without the real model or PyTorch. The maths reuses tools/dump_reference.py's
RoPE/attention helpers, which were validated against Hugging Face by that
script's attention self-check. Not a substitute for the real golden run.

  python tools/make_synthetic_checkpoint.py --out /tmp/fake-small --small
  python tools/make_synthetic_golden.py --model-dir /tmp/fake-small --out /tmp/fake-small/golden-math
  edgeforge verify-math --golden /tmp/fake-small/golden-math --model-dir /tmp/fake-small
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_reference import FileWriter, apply_rope, causal_gqa_attention, rope_cos_sin, write_json  # noqa: E402


def load_bf16_safetensors(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    base = 8 + n
    out = {}
    for name, e in header.items():
        if name == "__metadata__":
            continue
        b, end = e["data_offsets"]
        bits = np.frombuffer(raw[base + b:base + end], dtype="<u2").astype(np.uint32) << 16
        out[name] = bits.view(np.float32).astype(np.float64).reshape(e["shape"])
    return out


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return w * (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=7)
    args = ap.parse_args()

    cfg = json.loads((args.model_dir / "config.json").read_text())
    W = load_bf16_safetensors(args.model_dir / "model.safetensors")
    D, F, V, L = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"], cfg["num_hidden_layers"]
    H, KV = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd, eps, theta = D // H, cfg["rms_norm_eps"], cfg["rope_theta"]
    S = args.tokens
    rng = np.random.default_rng(0)
    ids = np.concatenate([[1], rng.integers(3, V, size=S - 1)]).astype(np.int32)
    p = "model.layers.0."

    embed = W["model.embed_tokens.weight"][ids]
    h = rmsnorm(embed, W[p + "input_layernorm.weight"], eps)
    q, k, v = h @ W[p + "self_attn.q_proj.weight"].T, h @ W[p + "self_attn.k_proj.weight"].T, h @ W[p + "self_attn.v_proj.weight"].T
    cos, sin = rope_cos_sin(np.arange(S), hd, theta)
    q_r = apply_rope(q.reshape(S, H, hd), cos.astype(np.float64), sin.astype(np.float64))
    k_r = apply_rope(k.reshape(S, KV, hd), cos.astype(np.float64), sin.astype(np.float64))
    ctx = causal_gqa_attention(q_r, k_r, v.reshape(S, KV, hd))
    attn_out = ctx @ W[p + "self_attn.o_proj.weight"].T
    resid = embed + attn_out
    post = rmsnorm(resid, W[p + "post_attention_layernorm.weight"], eps)
    gate, up = post @ W[p + "mlp.gate_proj.weight"].T, post @ W[p + "mlp.up_proj.weight"].T
    act = gate / (1.0 + np.exp(-gate)) * up
    mlp = act @ W[p + "mlp.down_proj.weight"].T
    out = resid + mlp
    hidden = np.stack([out] + [rng.standard_normal((S, D)) for _ in range(L - 1)])
    fnorm = rmsnorm(hidden[-1], W["model.norm.weight"], eps)
    logits = fnorm @ W["lm_head.weight"].T

    w = FileWriter(args.out / "prompts" / "p0_factual")
    w.i32("input_ids", ids, "synthetic ids")
    for key, arr in [("embed_all", embed), ("hidden_all", hidden), ("final_norm_all", fnorm),
                     ("final_norm_last", fnorm[-1]), ("logits_last", logits[-1]), ("rope_cos", cos), ("rope_sin", sin),
                     ("layer0/input_norm", h), ("layer0/q", q), ("layer0/k", k), ("layer0/v", v),
                     ("layer0/q_rope", q_r.reshape(S, D)), ("layer0/k_rope", k_r.reshape(S, KV * hd)),
                     ("layer0/attn_context", ctx), ("layer0/attn_out", attn_out), ("layer0/resid_after_attn", resid),
                     ("layer0/post_attn_norm", post), ("layer0/gate", gate), ("layer0/up", up),
                     ("layer0/mlp_act", act), ("layer0/mlp_out", mlp), ("layer0/out", out)]:
        w.f32(key, arr, "synthetic float64 reference")
    write_json(w.dir / "manifest.json", {"format": "edgeforge-golden-v1", "name": "p0_factual", "input_ids": ids.tolist(),
                                         "files": w.files, "greedy": {"ids": [int(np.argmax(logits[-1]))]}})
    write_json(args.out / "meta.json", {"format": "edgeforge-golden-v1", "config": cfg,
                                        "reference_dtype": "float64 (numpy, synthetic)",
                                        "versions": {"transformers": "none (numpy)"}})
    print(f"wrote synthetic golden for {S} tokens to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
