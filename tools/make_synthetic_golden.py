#!/usr/bin/env python3
"""EdgeForge - golden data for a SYNTHETIC checkpoint, computed in numpy float64.

Exercises `edgeforge verify-math --model-dir ...` and `edgeforge verify-forward`
(full forward + KV-cache decoding) without the real model or PyTorch. The maths reuses tools/dump_reference.py's
RoPE/attention helpers, which were validated against Hugging Face by that
script's attention self-check. Not a substitute for the real golden run.

  python tools/make_synthetic_checkpoint.py --out /tmp/fake-small --small
  python tools/make_synthetic_golden.py --model-dir /tmp/fake-small --out /tmp/fake-small/golden-math
  edgeforge verify-math --golden /tmp/fake-small/golden-math --model-dir /tmp/fake-small
  edgeforge verify-forward --golden /tmp/fake-small/golden-math --model-dir /tmp/fake-small
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


def forward_full(W: dict, cfg: dict, ids: np.ndarray) -> dict:
    """Whole-sequence float64 forward (no KV cache). Returns every intermediate we dump."""
    D, H, KV = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    L, hd, eps, theta = cfg["num_hidden_layers"], D // H, cfg["rms_norm_eps"], cfg["rope_theta"]
    S = len(ids)
    cos, sin = rope_cos_sin(np.arange(S), hd, theta)
    c64, s64 = cos.astype(np.float64), sin.astype(np.float64)
    x = W["model.embed_tokens.weight"][ids]
    out = {"embed": x.copy(), "cos": cos, "sin": sin, "hidden": []}
    for layer in range(L):
        p = f"model.layers.{layer}."
        h = rmsnorm(x, W[p + "input_layernorm.weight"], eps)
        q, k, v = h @ W[p + "self_attn.q_proj.weight"].T, h @ W[p + "self_attn.k_proj.weight"].T, h @ W[p + "self_attn.v_proj.weight"].T
        q_r, k_r = apply_rope(q.reshape(S, H, hd), c64, s64), apply_rope(k.reshape(S, KV, hd), c64, s64)
        ctx = causal_gqa_attention(q_r, k_r, v.reshape(S, KV, hd))
        attn_out = ctx @ W[p + "self_attn.o_proj.weight"].T
        resid = x + attn_out
        post = rmsnorm(resid, W[p + "post_attention_layernorm.weight"], eps)
        gate, up = post @ W[p + "mlp.gate_proj.weight"].T, post @ W[p + "mlp.up_proj.weight"].T
        act = gate / (1.0 + np.exp(-gate)) * up
        mlp = act @ W[p + "mlp.down_proj.weight"].T
        x = resid + mlp
        out["hidden"].append(x.copy())
        if layer == 0:
            out["layer0"] = {"input_norm": h, "q": q, "k": k, "v": v, "q_rope": q_r.reshape(S, D),
                             "k_rope": k_r.reshape(S, KV * hd), "attn_context": ctx, "attn_out": attn_out,
                             "resid_after_attn": resid, "post_attn_norm": post, "gate": gate, "up": up,
                             "mlp_act": act, "mlp_out": mlp, "out": x.copy()}
    out["final_norm"] = rmsnorm(x, W["model.norm.weight"], eps)
    head = W["model.embed_tokens.weight"] if cfg.get("tie_word_embeddings") else W["lm_head.weight"]
    out["logits"] = out["final_norm"] @ head.T
    return out


def write_prompt(root: Path, name: str, W: dict, cfg: dict, ids: np.ndarray, detail: bool, max_new: int,
                 decode_steps: int) -> None:
    f = forward_full(W, cfg, ids)
    w = FileWriter(root / "prompts" / name)
    w.i32("input_ids", ids, "synthetic ids")
    w.f32("embed_last", f["embed"][-1], "")
    w.f32("hidden_last", np.stack([h[-1] for h in f["hidden"]]), "")
    w.f32("final_norm_last", f["final_norm"][-1], "")
    w.f32("logits_last", f["logits"][-1], "")
    if detail:
        w.f32("embed_all", f["embed"], "")
        w.f32("hidden_all", np.stack(f["hidden"]), "")
        w.f32("final_norm_all", f["final_norm"], "")
        w.f32("logits_all", f["logits"], "")
        w.f32("rope_cos", f["cos"], "")
        w.f32("rope_sin", f["sin"], "")
        for key, arr in f["layer0"].items():
            w.f32("layer0/" + key, arr, "")
    # Greedy decoding by full recomputation (no cache): the C++ engine must match it via its KV cache.
    eos = cfg["eos_token_id"]
    seq, steps, decode = list(ids), [], []
    logits = f["logits"][-1]
    for step in range(max_new):
        order = np.argsort(-logits)
        tok = int(order[0])
        steps.append({"step": step, "token": tok, "margin_top1_top2": float(logits[order[0]] - logits[order[1]])})
        seq.append(tok)
        if tok == eos:
            break
        logits = forward_full(W, cfg, np.array(seq, dtype=np.int32))["logits"][-1]
        if len(decode) < decode_steps:
            decode.append(logits)
    if decode:
        w.f32("decode_logits", np.stack(decode), "")
    gen = [s["token"] for s in steps]
    write_json(w.dir / "manifest.json", {"format": "edgeforge-golden-v1", "name": name, "input_ids": ids.tolist(),
                                         "n_tokens": len(ids), "detail": detail, "files": w.files,
                                         "greedy": {"ids": gen, "max_new_tokens": max_new,
                                                    "stopped_on_eos": bool(gen and gen[-1] == eos), "steps": steps}})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=7)
    ap.add_argument("--max-new", type=int, default=6)
    args = ap.parse_args()

    cfg = json.loads((args.model_dir / "config.json").read_text())
    W = load_bf16_safetensors(args.model_dir / "model.safetensors")
    V = cfg["vocab_size"]
    rng = np.random.default_rng(0)
    p0 = np.concatenate([[1], rng.integers(3, V, size=args.tokens - 1)]).astype(np.int32)
    p1 = np.concatenate([[1], rng.integers(3, V, size=4)]).astype(np.int32)
    write_prompt(args.out, "p0_factual", W, cfg, p0, True, args.max_new, 2)
    write_prompt(args.out, "p1_short", W, cfg, p1, False, args.max_new, 2)
    write_json(args.out / "meta.json", {"format": "edgeforge-golden-v1", "config": cfg,
                                        "reference_dtype": "float64 (numpy, synthetic)",
                                        "versions": {"transformers": "none (numpy)"}})
    print(f"wrote synthetic golden (2 prompts, full forward + greedy) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
