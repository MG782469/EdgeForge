// EdgeForge - `edgeforge verify-forward`: the complete FP32 forward pass (all
// layers + KV-cache decoding) against PyTorch golden outputs.
#pragma once

#include <filesystem>
#include <string>

namespace ef {

struct VerifyForwardOptions {
  std::filesystem::path golden_dir;
  std::filesystem::path model_dir;
  std::string prompt;  // empty = every prompt under golden/prompts
};

// Returns 0 when every check passes, 1 otherwise.
int run_verify_forward(const VerifyForwardOptions& options);

}  // namespace ef
