#!/usr/bin/env python3
"""EdgeForge - write a FAKE TinyLlama-shaped checkpoint (random BF16 weights).

For testing the loader / verify pipeline end-to-end without downloading the real
model. Produces config.json + model.safetensors with the exact TinyLlama-1.1B
tensor names, shapes and dtype (2.2 GB), plus weight_probe.json in the same
format tools/dump_reference.py writes. The weights are random: this is NOT a
model and generates nothing meaningful.

  python tools/make_synthetic_checkpoint.py --out /tmp/fake-tinyllama            # full size
  python tools/make_synthetic_checkpoint.py --out /tmp/fake-small --small        # tiny dims
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_reference import PROBE_TENSORS, probe_entry  # noqa: E402

TINYLLAMA_CONFIG = {
    "architectures": ["LlamaForCausalLM"], "attention_bias": False, "attention_dropout": 0.0,
    "bos_token_id": 1, "eos_token_id": 2, "hidden_act": "silu", "hidden_size": 2048,
    "initializer_range": 0.02, "intermediate_size": 5632, "max_position_embeddings": 2048,
    "model_type": "llama", "num_attention_heads": 32, "num_hidden_layers": 22, "num_key_value_heads": 4,
    "pretraining_tp": 1, "rms_norm_eps": 1e-05, "rope_scaling": None, "rope_theta": 10000.0,
    "tie_word_embeddings": False, "torch_dtype": "bfloat16", "transformers_version": "4.35.0",
    "use_cache": True, "vocab_size": 32000,
}


def expected_tensors(c: dict) -> list[tuple[str, list[int]]]:
    D, F, V = c["hidden_size"], c["intermediate_size"], c["vocab_size"]
    KV = c["num_key_value_heads"] * (D // c["num_attention_heads"])
    out = [("model.embed_tokens.weight", [V, D])]
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}."
        out += [(p + "self_attn.q_proj.weight", [D, D]), (p + "self_attn.k_proj.weight", [KV, D]),
                (p + "self_attn.v_proj.weight", [KV, D]), (p + "self_attn.o_proj.weight", [D, D]),
                (p + "mlp.gate_proj.weight", [F, D]), (p + "mlp.up_proj.weight", [F, D]),
                (p + "mlp.down_proj.weight", [D, F]), (p + "input_layernorm.weight", [D]),
                (p + "post_attention_layernorm.weight", [D])]
    out.append(("model.norm.weight", [D]))
    if not c["tie_word_embeddings"]:
        out.append(("lm_head.weight", [V, D]))
    return out


def f32_to_bf16_bits(a: np.ndarray) -> np.ndarray:
    u = a.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)  # round to nearest even (no NaNs here)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--small", action="store_true", help="tiny dims instead of the real 1.1B shapes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = dict(TINYLLAMA_CONFIG)
    if args.small:
        cfg.update(hidden_size=64, intermediate_size=176, num_hidden_layers=2, num_attention_heads=4,
                   num_key_value_heads=2, vocab_size=100)
    tensors = expected_tensors(cfg)
    # HF writers order tensors by name; header keys follow the same order here.
    tensors.sort(key=lambda t: t[0])
    header: dict = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, shape in tensors:
        n = int(np.prod(shape))
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + 2 * n]}
        offset += 2 * n
    hjson = json.dumps(header, separators=(",", ":")).encode()
    hjson += b" " * ((8 - (8 + len(hjson)) % 8) % 8)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    rng = np.random.default_rng(args.seed)
    probes = []
    with open(args.out / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for name, shape in tensors:
            n = int(np.prod(shape))
            if name.endswith("norm.weight"):
                vals = 1.0 + 0.1 * rng.standard_normal(n, dtype=np.float32)
            else:
                vals = 0.02 * rng.standard_normal(n, dtype=np.float32)
            bits = f32_to_bf16_bits(vals)
            bits.astype("<u2").tofile(f)
            if name in PROBE_TENSORS:
                exact = (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
                probes.append(probe_entry(name, "BF16", shape, bits, exact))
    probe_dir = args.out / "golden"
    probe_dir.mkdir(exist_ok=True)
    probes.sort(key=lambda e: PROBE_TENSORS.index(e["name"]))
    (probe_dir / "weight_probe.json").write_text(json.dumps({
        "format": "edgeforge-weight-probe-v1", "source": "make_synthetic_checkpoint.py (numpy)",
        "file": "model.safetensors", "tensors": probes}, indent=1) + "\n", encoding="utf-8")
    total = 8 + len(hjson) + offset
    print(f"wrote {args.out}/model.safetensors ({total:,} bytes, {len(tensors)} tensors, "
          f"{offset // 2:,} params) + config.json + golden/weight_probe.json ({len(probes)} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
