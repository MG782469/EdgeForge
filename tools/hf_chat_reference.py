#!/usr/bin/env python3
"""EdgeForge - Stage 8, development-only: Hugging Face greedy chat replies, the reference for
`edgeforge verify-chat`.

For every case in tests/fixtures/chat_cases.json and every system prompt listed there ("none" and
the engine's default), this renders the conversation with the tokenizer's own chat template
(apply_chat_template, add_generation_prompt=True), tokenizes it with BOS exactly as the model
card's pipeline does, and generates greedily in FP32 with the KV cache. Generation stops only at
EOS or max_new_tokens: this is the raw model, with no stop strings or repetition rules. Multi-turn
cases continue with the reference's own replies.

Output: tests/golden/chat_generation.json with, per turn, the prompt ids, the generated ids (EOS
included when the model produced it), and at every step the top-1/top-2 logit margin and P(EOS).
PyTorch is only a test oracle; nothing here is used by the engine.

--study (EOS-threshold study, `edgeforge eos-study`): the cases of tests/fixtures/chat_eos_study.json
(300 new tokens) -> tests/golden/chat_eos_study.json. Runs of chat_generation.json that are already
complete (same model, every reply ended with EOS) are reused; the rest is generated, and wherever a
prompt is also in chat_generation.json the new tokens must start with the saved ones.

  python tools/hf_chat_reference.py --model-dir ~/edgeforge-data/tinyllama [--out-dir tests/golden]
  python tools/hf_chat_reference.py --model-dir ~/edgeforge-data/tinyllama --study
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dump_reference as dr  # noqa: E402  (shared: model loading, hashing, JSON writing)

FORMAT = "edgeforge-chat-golden-v1"
ROOT = Path(__file__).resolve().parent.parent


def render(tok, messages: list[dict]) -> tuple[str, list[int]]:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    add_special = not text.startswith(tok.bos_token or "\0")  # the TinyLlama template has no BOS
    ids = tok(text, add_special_tokens=add_special)["input_ids"]
    if ids[0] != tok.bos_token_id or ids.count(tok.bos_token_id) != 1:
        sys.exit(f"expected exactly one BOS at position 0, got {ids[:5]}...")
    return text, ids


def greedy(torch, model, tok, ids: list[int], max_new: int) -> dict:
    eos = tok.eos_token_id
    generated, margins, eos_p = [], [], []
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], dtype=torch.long), use_cache=True)
        past, logits = out.past_key_values, out.logits[0, -1].float()
        for _ in range(max_new):
            top = torch.topk(logits, 2)
            margins.append(round(float(top.values[0] - top.values[1]), 6))
            eos_p.append(round(float(torch.softmax(logits.double(), dim=-1)[eos]), 6))
            nxt = int(top.indices[0])
            generated.append(nxt)
            if nxt == eos:
                break
            out = model(input_ids=torch.tensor([[nxt]], dtype=torch.long), past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[0, -1].float()
    stopped = bool(generated) and generated[-1] == eos
    reply = generated[:-1] if stopped else generated
    return {
        "generated_ids": generated,
        "stopped_on_eos": stopped,
        "text": tok.decode(reply, skip_special_tokens=True),
        "margins": margins,
        "eos_p": eos_p,
    }


def find_reusable(old: dict | None, sha: str, sys_name: str, sys_text: str, case: dict, max_new: int):
    """A run of the regression golden that is complete for this case: same model, system text and
    turns, and every reply ended with EOS (so a larger token limit cannot change it)."""
    if not old or old.get("model_safetensors_sha256") != sha or old["systems"].get(sys_name) != sys_text:
        return None
    for r in old["runs"]:
        if r["case"] == case["name"] and r["system"] == sys_name and [t["user"] for t in r["turns"]] == case["turns"]:
            if all(t["stopped_on_eos"] and len(t["generated_ids"]) <= max_new for t in r["turns"]):
                return r
    return None


def old_turn(old: dict | None, sys_name: str, case_name: str, prompt_ids: list[int]):
    if not old:
        return None
    for r in old["runs"]:
        if r["case"] == case_name and r["system"] == sys_name:
            for t in r["turns"]:
                if t["prompt_ids"] == prompt_ids:
                    return t
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--study", action="store_true",
                    help="EOS-threshold study: cases from tests/fixtures/chat_eos_study.json -> chat_eos_study.json; "
                         "complete runs of chat_generation.json are reused, the rest generated")
    ap.add_argument("--cases", type=Path, help="default: tests/fixtures/chat_cases.json (chat_eos_study.json with --study)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "tests/golden")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = torch default)")
    args = ap.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    cases_path = args.cases or ROOT / ("tests/fixtures/chat_eos_study.json" if args.study else "tests/fixtures/chat_cases.json")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    max_new = int(cases["max_new_tokens"])
    out_name = "chat_eos_study.json" if args.study else "chat_generation.json"
    old = None
    if args.study and (args.out_dir / "chat_generation.json").exists():
        old = json.loads((args.out_dir / "chat_generation.json").read_text(encoding="utf-8"))

    torch, transformers, _ = dr.import_torch_stack()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    from transformers import AutoTokenizer

    sha = dr.sha256_file(model_dir / "model.safetensors")
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    dr.log(f"torch {torch.__version__} | transformers {transformers.__version__}; loading {model_dir} (fp32) ...")
    model, attn_impl = dr.load_model(torch, transformers, model_dir, "fp32")

    runs = []
    reused = generated = prefix_checks = 0
    t0 = time.time()
    for sys_name, sys_text in cases["systems"].items():
        for case in cases["cases"]:
            prev = find_reusable(old, sha, sys_name, sys_text, case, max_new) if args.study else None
            if prev is not None:
                runs.append({**prev, "kind": case["kind"], "source": "chat_generation.json (complete: every reply ended with EOS)"})
                reused += 1
                dr.log(f"{case['name']:22s} [{sys_name:7s}] reused from chat_generation.json")
                continue
            messages = [{"role": "system", "content": sys_text}] if sys_text else []
            turns = []
            for user in case["turns"]:
                messages.append({"role": "user", "content": user})
                formatted, ids = render(tok, messages)
                g = greedy(torch, model, tok, ids, max_new)
                turn = {"user": user, "formatted": formatted, "prompt_ids": ids, **g}
                o = old_turn(old, sys_name, case["name"], ids)
                if o is not None:  # same prompt in the regression golden: its tokens must be a prefix of ours
                    n = min(len(o["generated_ids"]), len(g["generated_ids"]))
                    same = o["generated_ids"][:n] == g["generated_ids"][:n]
                    turn["prefix_check"] = (f"first {n} tokens identical to chat_generation.json" if same
                                            else "DIFFERS from chat_generation.json")
                    prefix_checks += 1
                    if not same:
                        sys.exit(f"{case['name']}: greedy tokens differ from chat_generation.json for the same prompt")
                turns.append(turn)
                messages.append({"role": "assistant", "content": g["text"]})
                generated += 1
                n = len(g["generated_ids"]) - g["stopped_on_eos"]
                dr.log(f"{case['name']:22s} [{sys_name:7s}] {'EOS' if g['stopped_on_eos'] else 'no EOS'} "
                       f"after {n:3d} tok: {g['text'][:60]!r}")
            runs.append({"case": case["name"], "kind": case["kind"], "system": sys_name,
                         "source": "generated now", "turns": turns})

    out = {
        "format": FORMAT,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "model_safetensors_sha256": sha,
        "reference_dtype": "float32",
        "attn_implementation": attn_impl,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "max_new_tokens": max_new,
        "eos_token_id": tok.eos_token_id,
        "systems": cases["systems"],
        "runs": runs,
    }
    if args.study:
        out.update({"study": True, "cases_file": str(cases_path.relative_to(ROOT)) if cases_path.is_relative_to(ROOT)
                    else str(cases_path), "thresholds": cases["thresholds"], "decision_rule": cases["decision_rule"],
                    "cases": cases["cases"], "reused_runs": reused, "prefix_checks": prefix_checks})
    dr.write_json(args.out_dir / out_name, out)
    n = sum(len(r["turns"]) for r in runs)
    e = sum(t["stopped_on_eos"] for r in runs for t in r["turns"])
    dr.log(f"wrote {args.out_dir / out_name}: {n} replies ({generated} generated, {reused} runs reused, "
           f"{prefix_checks} prefix checks passed), {e} ended with EOS ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
