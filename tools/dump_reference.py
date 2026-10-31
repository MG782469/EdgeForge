#!/usr/bin/env python3
"""EdgeForge - development-only golden reference generator.

Runs TinyLlama through Hugging Face transformers (PyTorch, CPU, FP32 by default)
and writes reference data that the C++ engine is tested against in later stages:

  tests/golden/
    meta.json                 versions, sha256 of the checkpoint, self-check results
    weight_probe.json         sampled raw weight bits + full-tensor sums (official safetensors reader)
    tokenizer_cases.json      120+ strings -> token ids / pieces / decoded text
    chat_template_cases.json  conversations -> exact formatted prompt + ids
    prompts/<name>/manifest.json + *.f32 / *.i32   per-prompt activations, logits, greedy tokens

None of this is used by the runtime. PyTorch/transformers are only an offline
oracle (confirm with the organisers that this is allowed).

Usage:
  python tools/dump_reference.py --model-dir D:/edgeforge-data/tinyllama --out-dir tests/golden
  python tools/dump_reference.py --check-env --model-dir ...   # quick setup check, no model run
  python tools/dump_reference.py --selftest                     # numpy-only maths self-test
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

FORMAT_VERSION = "edgeforge-golden-v1"

PROMPTS = [
    ("p0_factual", "What is the capital of France?"),
    ("p1_technical", "Explain in two sentences how a CPU cache makes programs faster."),
    ("p2_code", "Write a Python function that checks whether a number is prime."),
    ("p3_hindi", "भारत की राजधानी क्या है? एक वाक्य में बताइए।"),
]

# Tensors whose raw bits are cross-checked against the C++ loader.
PROBE_TENSORS = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.10.mlp.gate_proj.weight",
    "model.layers.21.mlp.down_proj.weight",
    "model.norm.weight",
    "lm_head.weight",
]

_ZWJ = chr(0x200D)
_NBSP = chr(0x00A0)
_SP = chr(0x2581)  # SentencePiece word-boundary marker

TOKENIZER_CASES = [
    "", " ", "  ", "   ", "a", "A", "I", "hello", "Hello", "Hello world", " Hello world", "Hello  world",
    "Hello world ", "  leading spaces", "trailing spaces  ", "Hello\nworld", "\n", "\n\n", "\n\n\n", "\t",
    "tab\tseparated\tvalues", "\r\n", "line one\r\nline two", "  \n  ", "a\n b\n  c",
    "12345", "3.14159", "1,000,000", "2024-10-31", "100%", "$9.99", "x = 42", "007", "1e-5",
    "Hello, world! How are you?", "Wait... what?!", "\"quoted\" and 'single'", "(parentheses) [brackets] {braces}",
    "semi;colon:colon", "a-b_c.d/e\\f", "@user #hashtag", "email@example.com", "https://example.com/path?q=1&r=2",
    "def f(x):\n    return x * 2\n", "for (int i = 0; i < n; ++i) {\n    sum += a[i];\n}",
    "#include <stdio.h>", "x += 1; // comment", "SELECT * FROM users WHERE id = 1;", '{"key": "value", "n": [1, 2]}',
    "<html><body>Hi</body></html>", "if a != b and not c: pass", "    indented four", "\tindented tab",
    "😀", "I ❤️ NLP", "👍🏽", "🇮🇳", "a" + _ZWJ + "b", "non" + _NBSP + "breaking",
    "नमस्ते", "भारत की राजधानी क्या है?", "मैं ठीक हूँ, धन्यवाद।", "bhai kaise ho", "Hello नमस्ते 😀 123",
    "你好，世界", "こんにちは", "안녕하세요", "مرحبا بالعالم", "Привет, мир", "Γειά σου", "שלום",
    "café naïve résumé", "Straße", "ñandú", "Ångström", "∑ x² = ∞", "α β γ δ", "√2 ≈ 1.414", "€ £ ¥ ₹",
    "</s>", "<s>", "<unk>", "<s>Hello</s>", "Hello</s>", "</s>\n", "<|user|>", "<|assistant|>", "<|system|>",
    "<|user|>\nHi</s>\n<|assistant|>\n", "<0x0A>", _SP, _SP + "hello", "▁▁", "<pad>", "[INST]",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "the the the the the", "ababababab", "..........", "----------",
    "!!!", "???", "Mr. Smith's dog's bone", "don't won't can't", "U.S.A.", "e.g. i.e. etc.",
    "CamelCaseIdentifier", "snake_case_identifier", "SCREAMING_CASE", "ALLCAPS", "MiXeD cAsE",
    "TinyLlama", "EdgeForge", "BIT Mesra", "Thapar", "Ranchi", "quantization", "dequantise",
    "The quick brown fox jumps over the lazy dog.", chr(7) + "bell", "zero" + chr(0) + "byte",
    "Once upon a time, there was a little robot who wanted to learn how to paint.",
]

CHAT_CASES = [
    [{"role": "user", "content": "Hi"}],
    [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is 2+2?"}],
    [
        {"role": "user", "content": "Name a fruit."},
        {"role": "assistant", "content": "Apple."},
        {"role": "user", "content": "Another one?"},
    ],
    [{"role": "user", "content": "  leading spaces and trailing  "}],
    [{"role": "user", "content": "नमस्ते! 😀"}],
]


# ----------------------------------------------------------------------------- small utils

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def available_ram_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullAvailPhys)
        except Exception:
            pass
    return None


class FileWriter:
    """Writes little-endian raw tensors and records them in a manifest."""

    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self.files: dict[str, dict] = {}

    def f32(self, name: str, arr, desc: str) -> None:
        a = np.ascontiguousarray(np.asarray(arr, dtype="<f4"))
        fname = name + ".f32"
        (self.dir / fname).parent.mkdir(parents=True, exist_ok=True)
        a.tofile(self.dir / fname)
        self.files[name] = {"file": fname, "dtype": "f32", "shape": list(a.shape), "description": desc}

    def i32(self, name: str, arr, desc: str) -> None:
        a = np.ascontiguousarray(np.asarray(arr, dtype="<i4"))
        fname = name + ".i32"
        a.tofile(self.dir / fname)
        self.files[name] = {"file": fname, "dtype": "i32", "shape": list(a.shape), "description": desc}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")


# ----------------------------------------------------------------------------- weight probe (no torch)

def sample_indices(numel: int, seed: int = 1234) -> list[int]:
    """First 8, middle 8, last 8 and 8 pseudo-random flat indices."""
    idx = set(range(min(8, numel)))
    mid = numel // 2
    idx.update(i for i in range(mid, min(mid + 8, numel)))
    idx.update(range(max(0, numel - 8), numel))
    rng = np.random.default_rng(seed + numel % 9973)
    if numel > 0:
        idx.update(int(i) for i in rng.integers(0, numel, size=8))
    return sorted(idx)


def probe_entry(name: str, dtype: str, shape: list[int], bits: np.ndarray, values_f64: np.ndarray) -> dict:
    """bits: flat unsigned raw bit patterns; values_f64: the same elements as float64."""
    numel = int(bits.size)
    idx = sample_indices(numel)
    return {
        "name": name,
        "dtype": dtype,
        "shape": [int(s) for s in shape],
        "numel": numel,
        "samples": [[int(i), int(bits[i])] for i in idx],
        "first8_f32": [float(v) for v in values_f64[:8]],
        "sum_f64": float(np.sum(values_f64, dtype=np.float64)),
        "sum_abs_f64": float(np.sum(np.abs(values_f64), dtype=np.float64)),
    }


# ----------------------------------------------------------------------------- reference maths (numpy)

def rope_cos_sin(positions: np.ndarray, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Hugging Face Llama RoPE tables (half-split / rotate_half convention), float32."""
    inv_freq = (1.0 / (np.float32(theta) ** (np.arange(0, head_dim, 2, dtype=np.int64).astype(np.float32) / np.float32(head_dim)))).astype(np.float32)
    freqs = np.outer(positions.astype(np.float32), inv_freq).astype(np.float32)  # [S, hd/2]
    emb = np.concatenate([freqs, freqs], axis=-1)  # [S, hd]: dim i pairs with dim i + hd/2
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rotate_half(x: np.ndarray) -> np.ndarray:
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """x: [S, n_heads, head_dim]; cos/sin: [S, head_dim]."""
    return x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]


def causal_gqa_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """q: [S, H, hd], k/v: [S, KV, hd] (already RoPE'd). Returns [S, H*hd] in float64.
    Query head h reads KV head h // (H // KV) - the Hugging Face repeat_kv mapping."""
    S, H, hd = q.shape
    KV = k.shape[1]
    group = H // KV
    out = np.zeros((S, H * hd), dtype=np.float64)
    mask = np.triu(np.ones((S, S), dtype=bool), k=1)
    for h in range(H):
        kv = h // group
        scores = (q[:, h, :].astype(np.float64) @ k[:, kv, :].astype(np.float64).T) / np.sqrt(hd)
        scores[mask] = -np.inf
        scores -= scores.max(axis=1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(axis=1, keepdims=True)
        out[:, h * hd:(h + 1) * hd] = p @ v[:, kv, :].astype(np.float64)
    return out


def selftest() -> int:
    """Checks the numpy reference maths against naive per-element loops."""
    rng = np.random.default_rng(0)
    S, H, KV, hd, theta = 5, 4, 2, 8, 10000.0
    cos, sin = rope_cos_sin(np.arange(S), hd, theta)
    x = rng.standard_normal((S, H, hd)).astype(np.float32)
    got = apply_rope(x, cos, sin)
    want = np.zeros_like(x)
    for s in range(S):
        for h in range(H):
            for i in range(hd // 2):
                ang = s / theta ** (2 * i / hd)
                a, b = x[s, h, i], x[s, h, i + hd // 2]  # half-split pairing (NOT adjacent pairs)
                want[s, h, i] = a * np.cos(ang) - b * np.sin(ang)
                want[s, h, i + hd // 2] = b * np.cos(ang) + a * np.sin(ang)
    err_rope = float(np.max(np.abs(got - want)))
    q = rng.standard_normal((S, H, hd))
    k = rng.standard_normal((S, KV, hd))
    v = rng.standard_normal((S, KV, hd))
    got_attn = causal_gqa_attention(q, k, v)
    want_attn = np.zeros((S, H * hd))
    for s in range(S):
        for h in range(H):
            kv = h // (H // KV)
            sc = np.array([q[s, h] @ k[t, kv] / np.sqrt(hd) for t in range(s + 1)])
            p = np.exp(sc - sc.max())
            p /= p.sum()
            want_attn[s, h * hd:(h + 1) * hd] = sum(p[t] * v[t, kv] for t in range(s + 1))
    err_attn = float(np.max(np.abs(got_attn - want_attn)))
    probe = probe_entry("t", "BF16", [3, 4], np.arange(12, dtype=np.uint16), np.arange(12, dtype=np.float64))
    ok = err_rope < 1e-5 and err_attn < 1e-12 and probe["sum_f64"] == 66.0 and len(probe["samples"]) == 12
    print(f"selftest: rope max err {err_rope:.2e}, attention max err {err_attn:.2e}, probe ok "
          f"{probe['sum_f64'] == 66.0} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ----------------------------------------------------------------------------- torch part

def import_torch_stack():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError as e:
        sys.exit(f"missing dependency: {e.name}. Install with:\n"
                 f"  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                 f"  python -m pip install transformers safetensors numpy")
    import torch
    import transformers
    import safetensors
    return torch, transformers, safetensors


def load_model(torch, transformers, model_dir: Path, dtype_name: str):
    from transformers import AutoModelForCausalLM

    dtype = torch.float32 if dtype_name == "fp32" else torch.bfloat16
    ver = tuple(int(p) for p in transformers.__version__.split(".")[:2] if p.isdigit())
    kwargs = {"local_files_only": True, "low_cpu_mem_usage": True}
    kwargs["dtype" if ver >= (4, 56) else "torch_dtype"] = dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), attn_implementation="eager", **kwargs)
        attn_impl = "eager"
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
        attn_impl = "default"
    model = model.to(dtype)
    model.eval()
    if next(model.parameters()).dtype != dtype:
        sys.exit("model did not load in the requested dtype")
    return model, attn_impl


def to_np(t) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def register_hooks(model, detail: bool, store: dict):
    base = model.model
    handles = []

    def out_hook(name):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            store[name] = to_np(t)[0]
        return hook

    def in_hook(name):
        def hook(_mod, args):
            store[name] = to_np(args[0])[0]
        return hook

    handles.append(base.embed_tokens.register_forward_hook(out_hook("embed")))
    for i, layer in enumerate(base.layers):
        handles.append(layer.register_forward_hook(out_hook(f"layer_{i}")))
    handles.append(base.norm.register_forward_hook(out_hook("final_norm")))
    if detail:
        l0 = base.layers[0]
        handles += [
            l0.input_layernorm.register_forward_hook(out_hook("l0_input_norm")),
            l0.self_attn.q_proj.register_forward_hook(out_hook("l0_q")),
            l0.self_attn.k_proj.register_forward_hook(out_hook("l0_k")),
            l0.self_attn.v_proj.register_forward_hook(out_hook("l0_v")),
            l0.self_attn.o_proj.register_forward_pre_hook(in_hook("l0_attn_context")),
            l0.self_attn.o_proj.register_forward_hook(out_hook("l0_attn_out")),
            l0.post_attention_layernorm.register_forward_pre_hook(in_hook("l0_resid_after_attn")),
            l0.post_attention_layernorm.register_forward_hook(out_hook("l0_post_attn_norm")),
            l0.mlp.gate_proj.register_forward_hook(out_hook("l0_gate")),
            l0.mlp.up_proj.register_forward_hook(out_hook("l0_up")),
            l0.mlp.down_proj.register_forward_pre_hook(in_hook("l0_mlp_act")),
            l0.mlp.register_forward_hook(out_hook("l0_mlp_out")),
        ]
    return handles


def run_prompt(torch, model, tok, cfg: dict, name: str, user_text: str, out_root: Path, detail: bool,
               max_new: int, decode_logit_steps: int, tol: float) -> dict:
    eos_id = tok.eos_token_id
    bos_id = tok.bos_token_id
    messages = [{"role": "user", "content": user_text}]
    formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # The TinyLlama template has no BOS; tokenising with add_special_tokens=True adds exactly one.
    add_special = not formatted.startswith(tok.bos_token or "\0")
    ids = tok(formatted, add_special_tokens=add_special)["input_ids"]
    if ids[0] != bos_id or ids.count(bos_id) != 1:
        sys.exit(f"{name}: expected exactly one BOS at position 0, got {ids[:5]}...")
    S = len(ids)
    L, D = cfg["num_hidden_layers"], cfg["hidden_size"]
    H, KV = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = D // H

    store: dict[str, np.ndarray] = {}
    handles = register_hooks(model, detail, store)
    input_ids = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    for h in handles:
        h.remove()
    logits_all = to_np(out.logits)[0]  # [S, V]

    w = FileWriter(out_root / "prompts" / name)
    w.i32("input_ids", ids, "prompt token ids (BOS + chat template)")
    w.f32("embed_last", store["embed"][-1], "token embedding at the last position")
    w.f32("hidden_last", np.stack([store[f"layer_{i}"][-1] for i in range(L)]), "[layer, D] decoder-layer outputs, last position")
    w.f32("final_norm_last", store["final_norm"][-1], "final RMSNorm output, last position")
    w.f32("logits_last", logits_all[-1], "next-token logits after the prompt")

    self_check = None
    if detail:
        w.f32("embed_all", store["embed"], "[S, D] token embeddings")
        w.f32("hidden_all", np.stack([store[f"layer_{i}"] for i in range(L)]), "[layer, S, D] decoder-layer outputs, all positions")
        w.f32("final_norm_all", store["final_norm"], "[S, D] final RMSNorm output")
        w.f32("logits_all", logits_all, "[S, V] logits at every prompt position")
        cos, sin = rope_cos_sin(np.arange(S), hd, float(cfg.get("rope_theta", 10000.0)))
        w.f32("rope_cos", cos, "[S, head_dim] RoPE cos table (half-split convention)")
        w.f32("rope_sin", sin, "[S, head_dim] RoPE sin table")
        q = store["l0_q"].reshape(S, H, hd)
        k = store["l0_k"].reshape(S, KV, hd)
        v = store["l0_v"].reshape(S, KV, hd)
        q_r, k_r = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        for key, desc in [
            ("l0_input_norm", "[S, D] layer 0 input RMSNorm output"),
            ("l0_q", "[S, D] q_proj output (before RoPE)"),
            ("l0_k", "[S, KV*hd] k_proj output (before RoPE)"),
            ("l0_v", "[S, KV*hd] v_proj output"),
            ("l0_attn_context", "[S, D] attention output before o_proj"),
            ("l0_attn_out", "[S, D] o_proj output"),
            ("l0_resid_after_attn", "[S, D] residual stream after attention"),
            ("l0_post_attn_norm", "[S, D] post-attention RMSNorm output"),
            ("l0_gate", "[S, F] gate_proj output"),
            ("l0_up", "[S, F] up_proj output"),
            ("l0_mlp_act", "[S, F] silu(gate) * up (down_proj input)"),
            ("l0_mlp_out", "[S, D] MLP output"),
        ]:
            w.f32(f"layer0/{key[3:]}", store[key], desc)
        w.f32("layer0/q_rope", q_r.reshape(S, D), "[S, D] q after RoPE (computed here, validated below)")
        w.f32("layer0/k_rope", k_r.reshape(S, KV * hd), "[S, KV*hd] k after RoPE")
        w.f32("layer0/out", store["layer_0"], "[S, D] layer 0 output")
        # Self-check: our RoPE + GQA + causal softmax must reproduce HF's attention context.
        ours = causal_gqa_attention(q_r, k_r, v)
        ref = store["l0_attn_context"].astype(np.float64)
        diff = float(np.max(np.abs(ours - ref)))
        scale = float(np.max(np.abs(ref)))
        # Same check with the WRONG (adjacent-pair) RoPE, to show the test can tell them apart.
        def adjacent_rope(x):
            y = x.reshape(S, x.shape[1], hd // 2, 2)
            ang = np.outer(np.arange(S), 1.0 / (float(cfg.get("rope_theta", 10000.0)) ** (np.arange(0, hd, 2) / hd)))
            c, s_ = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
            a, b = y[..., 0], y[..., 1]
            return np.stack([a * c - b * s_, a * s_ + b * c], axis=-1).reshape(x.shape)
        wrong = float(np.max(np.abs(causal_gqa_attention(adjacent_rope(q), adjacent_rope(k), v) - ref)))
        self_check = {"max_abs_diff": diff, "max_abs_ref": scale, "tolerance": tol, "passed": diff <= tol,
                      "adjacent_pair_rope_diff": wrong}
        log(f"  attention self-check: max|ours-HF| = {diff:.3e} (tol {tol:g}) -> {'PASS' if diff <= tol else 'FAIL'};"
            f" wrong adjacent-pair RoPE would give {wrong:.3e}")

    # Greedy decoding with the KV cache: pure argmax, no generation_config surprises.
    steps, generated, decode_logits = [], [], []
    with torch.no_grad():
        past = out.past_key_values
        cur = out.logits[0, -1].float()
        for step in range(max_new):
            top = torch.topk(cur, 5)
            tok_id = int(torch.argmax(cur))
            steps.append({
                "step": step,
                "token": tok_id,
                "piece": tok.convert_ids_to_tokens(tok_id),
                "top5": [[int(i), float(vv)] for i, vv in zip(top.indices, top.values)],
                "margin_top1_top2": float(top.values[0] - top.values[1]),
            })
            generated.append(tok_id)
            if tok_id == eos_id:
                break
            o = model(input_ids=torch.tensor([[tok_id]], dtype=torch.long), past_key_values=past, use_cache=True)
            past = o.past_key_values
            cur = o.logits[0, -1].float()
            if len(decode_logits) < decode_logit_steps:
                decode_logits.append(cur.cpu().numpy())
    if decode_logits:
        w.f32("decode_logits", np.stack(decode_logits),
              "[n, V] logits after feeding greedy tokens 1..n through the KV cache")

    manifest = {
        "format": FORMAT_VERSION,
        "name": name,
        "prompt": user_text,
        "formatted": formatted,
        "add_special_tokens": add_special,
        "input_ids": ids,
        "n_tokens": S,
        "detail": detail,
        "files": w.files,
        "greedy": {
            "ids": generated,
            "text": tok.decode(generated, skip_special_tokens=True),
            "text_with_special": tok.decode(generated, skip_special_tokens=False),
            "stopped_on_eos": bool(generated and generated[-1] == eos_id),
            "max_new_tokens": max_new,
            "steps": steps,
        },
        "attention_self_check": self_check,
    }
    write_json(w.dir / "manifest.json", manifest)
    log(f"  {name}: {S} prompt tokens, {len(generated)} generated -> {manifest['greedy']['text'][:70]!r}")
    return manifest


def dump_tokenizer(tok, out_dir: Path) -> None:
    cases = []
    for s in TOKENIZER_CASES:
        ids = tok.encode(s, add_special_tokens=False)
        cases.append({
            "text": s,
            "ids": ids,
            "ids_default": tok.encode(s),  # with the tokenizer's default special tokens (BOS)
            "pieces": tok.convert_ids_to_tokens(ids),
            "decoded": tok.decode(ids),
            "decoded_skip_special": tok.decode(ids, skip_special_tokens=True),
        })
    write_json(out_dir / "tokenizer_cases.json", {
        "format": FORMAT_VERSION,
        "tokenizer_class": type(tok).__name__,
        "legacy": getattr(tok, "legacy", None),
        "vocab_size": tok.vocab_size,
        "bos_id": tok.bos_token_id, "eos_id": tok.eos_token_id, "unk_id": tok.unk_token_id,
        "add_bos_token": getattr(tok, "add_bos_token", None), "add_eos_token": getattr(tok, "add_eos_token", None),
        "cases": cases,
    })
    chats = []
    for conv in CHAT_CASES:
        for gen in (True, False):
            text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=gen)
            chats.append({"messages": conv, "add_generation_prompt": gen, "text": text,
                          "ids": tok(text, add_special_tokens=True)["input_ids"]})
    write_json(out_dir / "chat_template_cases.json", {
        "format": FORMAT_VERSION, "chat_template": tok.chat_template, "cases": chats,
        "note": "ids = BOS + tokenised text (add_special_tokens=True); the template itself adds no BOS",
    })
    log(f"tokenizer: {len(cases)} cases, {len(chats)} chat-template cases")


def dump_weight_probe(torch, safetensors, weights: Path, out_dir: Path) -> None:
    from safetensors import safe_open

    entries = []
    names = {"torch.bfloat16": "BF16", "torch.float16": "F16", "torch.float32": "F32"}
    with safe_open(str(weights), framework="pt") as f:
        keys = set(f.keys())
        for name in PROBE_TENSORS:
            if name not in keys:
                continue
            t = f.get_tensor(name).contiguous()
            flat = t.reshape(-1)
            if t.dtype in (torch.bfloat16, torch.float16):
                bits = flat.view(torch.int16).numpy().view(np.uint16)
            elif t.dtype == torch.float32:
                bits = flat.view(torch.int32).numpy().view(np.uint32)
            else:
                continue
            entries.append(probe_entry(name, names[str(t.dtype)], list(t.shape), bits,
                                       flat.to(torch.float64).numpy()))
    write_json(out_dir / "weight_probe.json", {
        "format": "edgeforge-weight-probe-v1",
        "source": f"safetensors.safe_open (safetensors {safetensors.__version__})",
        "file": weights.name,
        "tensors": entries,
    })
    log(f"weight probe: {len(entries)} tensors, {sum(len(e['samples']) for e in entries)} sampled values")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Hindi/emoji on Windows consoles
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, help="folder with config.json, model.safetensors, tokenizer files")
    ap.add_argument("--out-dir", type=Path, default=Path("tests/golden"))
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                    help="reference compute dtype (fp32 strongly recommended; bf16 only if RAM is short)")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--decode-logit-steps", type=int, default=4, help="full logits saved for the first N decode steps")
    ap.add_argument("--detail-prompt", default="p0_factual", help="prompt that gets all-position + layer-0 dumps")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = torch default)")
    ap.add_argument("--skip-sha256", action="store_true", help="do not hash model.safetensors (saves ~10 s)")
    ap.add_argument("--check-env", action="store_true", help="only check dependencies, files and RAM")
    ap.add_argument("--selftest", action="store_true", help="numpy-only maths self-test (no torch needed)")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.model_dir is None:
        ap.error("--model-dir is required")
    model_dir = args.model_dir.expanduser().resolve()
    needed = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
    missing = [n for n in needed if not (model_dir / n).exists()]
    if missing:
        sys.exit(f"missing in {model_dir}: {', '.join(missing)}\n"
                 f"download with: hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir {model_dir}")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never silently download anything
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    n_params = 1_100_048_384
    bytes_per = 4 if args.dtype == "fp32" else 2
    need = n_params * bytes_per + (1 << 30)
    avail = available_ram_bytes()
    log(f"model dir {model_dir}; reference dtype {args.dtype}; needs ~{need / 2**30:.1f} GiB RAM, "
        f"available {'?' if avail is None else f'{avail / 2**30:.1f} GiB'}")
    if avail is not None and avail < need:
        print(f"WARNING: probably not enough free RAM for a {args.dtype} reference. Close other apps, or use\n"
              f"         --dtype bf16 (labelled lower-precision reference; looser tolerances later).", flush=True)

    torch, transformers, safetensors = import_torch_stack()
    log(f"python {platform.python_version()} | torch {torch.__version__} | transformers {transformers.__version__} | "
        f"safetensors {safetensors.__version__} | numpy {np.__version__}")
    if args.check_env:
        print("check-env: OK (dependencies import, model files present)")
        return 0
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = model_dir / "model.safetensors"
    model_sha = None
    if not args.skip_sha256:
        log("hashing model.safetensors (sha256) ...")
        model_sha = sha256_file(weights)
        log(f"sha256 {model_sha}")

    dump_weight_probe(torch, safetensors, weights, out_dir)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    dump_tokenizer(tok, out_dir)

    log("loading model ...")
    t0 = time.time()
    model, attn_impl = load_model(torch, transformers, model_dir, args.dtype)
    log(f"loaded in {time.time() - t0:.1f} s (attention: {attn_impl}, torch threads {torch.get_num_threads()})")

    tol = 1e-3 if args.dtype == "fp32" else 5e-2
    manifests = []
    for name, text in PROMPTS:
        log(f"prompt {name} ...")
        manifests.append(run_prompt(torch, model, tok, cfg, name, text, out_dir, name == args.detail_prompt,
                                    args.max_new_tokens, args.decode_logit_steps, tol))

    checks = [m["attention_self_check"] for m in manifests if m["attention_self_check"]]
    meta = {
        "format": FORMAT_VERSION,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "model_dir": str(model_dir),
        "model_safetensors_sha256": model_sha,
        "config_sha256": sha256_file(model_dir / "config.json"),
        "reference_dtype": "float32" if args.dtype == "fp32" else "bfloat16",
        "attn_implementation": attn_impl,
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "safetensors": safetensors.__version__,
            "numpy": np.__version__,
        },
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "config": cfg,
        "prompts": [m["name"] for m in manifests],
        "detail_prompt": args.detail_prompt,
        "attention_self_check": checks[0] if checks else None,
        "binary_format": "raw little-endian, C order; *.f32 = float32, *.i32 = int32; shapes in each manifest.json",
    }
    write_json(out_dir / "meta.json", meta)
    ok = all(c["passed"] for c in checks)
    log(f"done -> {out_dir.resolve()}  (attention self-check: {'PASS' if ok else 'FAIL'})")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
