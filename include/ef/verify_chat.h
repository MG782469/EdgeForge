// EdgeForge - Stage 8: `edgeforge verify-chat`, chat replies against Hugging Face generation.
//
// Reads tests/golden/chat_generation.json (tools/hf_chat_reference.py: PyTorch FP32 greedy
// generation of the cases in tests/fixtures/chat_cases.json, with and without the default system
// prompt, multi-turn cases replayed with the reference's own replies) and runs three parts:
//   1. raw replay: our prompt ids must equal Hugging Face's, and the greedy reply ids too (BF16
//      engine: identical, or a first difference only where the reference's top-2 margin is a
//      near-tie; the 4-bit engine's agreement is reported). Stops: EOS and max-new only.
//   2. how the raw model ends its replies (EOS, max-new, role markers, repetition, peak P(EOS)) -
//      a property of the model, reported, so runtime and model behaviour are kept apart;
//   3. the default chat settings (system prompt, role-marker and repetition stops) on the same
//      cases: how and where each reply ends. Checked: no role marker ever reaches the reply, and
//      the repetition stop never cuts short a reply that the model ends with EOS on its own.
#pragma once

#include <filesystem>

namespace ef {

class ThreadPool;

struct VerifyChatOptions {
  std::filesystem::path golden_dir;     // folder with chat_generation.json
  std::filesystem::path model_dir;      // BF16 engine (and tokenizer, unless tokenizer_dir)
  std::filesystem::path qmodel;         // or: the 4-bit engine (.efq)
  std::filesystem::path tokenizer_dir;  // needed with qmodel
  ThreadPool* pool = nullptr;
};

int run_verify_chat(const VerifyChatOptions& o);

}  // namespace ef
