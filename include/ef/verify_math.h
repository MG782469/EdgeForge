// EdgeForge - `edgeforge verify-math`: checks every Stage 2 kernel (and a chained
// layer 0, final norm and lm_head) against PyTorch golden outputs.
#pragma once

#include <filesystem>
#include <string>

namespace ef {

struct VerifyMathOptions {
  std::filesystem::path golden_dir;  // tests/golden
  std::filesystem::path model_dir;   // empty = weight-free checks only
  std::filesystem::path bench_json;  // optional membench JSON, for % of peak bandwidth
  std::string prompt = "p0_factual";
};

// Returns 0 when every executed check passes, 1 otherwise.
int run_verify_math(const VerifyMathOptions& options);

}  // namespace ef
