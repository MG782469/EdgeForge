// EdgeForge - `edgeforge verify-tokenizer`: the C++ tokenizer, chat template and
// decoder against Hugging Face golden data; with a model, also end-to-end
// text -> tokens -> engine -> text against the PyTorch greedy outputs.
#pragma once

#include <filesystem>

namespace ef {

struct VerifyTokenizerOptions {
  std::filesystem::path golden_dir;
  std::filesystem::path tokenizer_dir;
  std::filesystem::path model_dir;  // empty = skip end-to-end generation
};

int run_verify_tokenizer(const VerifyTokenizerOptions& options);

}  // namespace ef
