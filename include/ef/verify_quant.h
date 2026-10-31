// EdgeForge - Stage 5 checks: `edgeforge verify-quant` and `edgeforge quant-analyze`.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "ef/quant.h"

namespace ef {

// One layer weight matrix evaluated on real activations: y = W x with the stored
// (dequantized) weights vs the exact BF16 weights, both in FP32 math.
struct GemvCheck {
  std::string tensor;
  QType type = QType::F32;
  double cosine = 1.0;        // min over prompts of the cosine over all of that prompt's outputs
  double rel_l2 = 0.0;        // ||y_q - y|| / ||y|| on that prompt
  double token_cosine = 1.0;  // worst single position over all prompts (informational)
  std::string prompt;         // prompt that gave `cosine`
  std::size_t prompts = 0, positions = 0;
};

// A tensor whose type comes from an override, against the type the uniform policy would give.
struct OverrideCheck {
  std::string tensor;
  QType stored = QType::F32, baseline = QType::F32;
  double rmse_stored = 0.0, rmse_baseline = 0.0;     // weight reconstruction rel RMSE
  double cos_stored = -1.0, cos_baseline = -1.0;     // GEMV cosine; -1 = not measured
  std::int64_t extra_bytes = 0;                      // stored bytes - baseline bytes
};

// A tensor that fails a bound, re-measured at the next precision that passes.
struct Promotion {
  std::string tensor, reason;
  QType from = QType::Q4_0, to = QType::Q8_0;
  double rmse_from = 0.0, rmse_to = 0.0;
  double cos_from = -1.0, cos_to = -1.0;  // -1 = not measured
  std::int64_t extra_bytes = 0;
};

struct VerifyQuantReport {
  int passed = 0, failed = 0;
  std::vector<std::string> failed_checks;
  double activation_rel_err = -1.0;       // re-derived layer outputs vs golden; < 0 = not run
  std::vector<std::string> prompts_used;  // prompts that contributed activations
  std::vector<GemvCheck> gemv;            // every layer matrix (empty without golden activations)
  std::vector<OverrideCheck> overrides;
  std::vector<Promotion> promotions;
  std::string suggested_override;         // --override value fixing every failure ("" = none needed)
};

struct VerifyQuantOptions {
  std::filesystem::path qmodel;     // .efq file
  std::filesystem::path model_dir;  // source checkpoint (model.safetensors + config.json)
  std::filesystem::path golden_dir; // optional: activation-level checks
  unsigned threads = 0;
  VerifyQuantReport* report = nullptr;  // optional: structured results (tests)
};
int run_verify_quant(const VerifyQuantOptions& o);

struct QuantAnalyzeOptions {
  std::filesystem::path model_dir;
  std::filesystem::path golden_dir;  // optional: output error on real layer-0 activations
};
int run_quant_analyze(const QuantAnalyzeOptions& o);

}  // namespace ef
