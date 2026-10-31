// EdgeForge - Stage 5: model quantizer (model.safetensors BF16 -> .efq).
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "ef/qfile.h"
#include "ef/safetensors.h"

namespace ef {

struct QuantizeOptions {
  QType layers = QType::Q4_0;   // attention + MLP matrices (the bandwidth-critical 88%)
  QType lm_head = QType::Q8_0;  // output projection: higher precision protects the logits
  QType embed = QType::BF16;    // only one row is read per token, so it costs no bandwidth
  // Per-tensor exceptions to the three defaults above (see QuantOverride). The library never
  // adds model-specific exceptions on its own: empty = uniform. Named, measured sets live in
  // named_policies() and are chosen explicitly (or by the CLI's architecture-guarded "auto").
  std::vector<QuantOverride> overrides;
  std::string policy_name;      // label written to the header ("" = "uniform" / "custom")
  bool search = true;           // per-block MSE scale search for Q4
  unsigned threads = 0;         // 0 = all hardware threads

  QuantPolicy policy() const {
    return QuantPolicy{layers, lm_head, embed, overrides,
                       !policy_name.empty() ? policy_name : overrides.empty() ? "uniform" : "custom"};
  }
};

// ---- named precision policies (version-controlled, measured on a specific checkpoint) ----------

struct NamedPolicy {
  std::string name;
  std::string measured_on;               // checkpoint the overrides were measured on
  std::vector<QuantOverride> overrides;  // each one fixed a failed verify-quant bound
  // Architecture guard: the policy only applies to a config with exactly these numbers
  // (hidden, intermediate, layers, heads, kv heads, vocab; untied). All zero = any model.
  std::int64_t hidden = 0, intermediate = 0, layers = 0, heads = 0, kv_heads = 0, vocab = 0;
  bool matches(const ModelConfig& c) const noexcept;
};

// "uniform" (no exceptions, any model) and "tinyllama-1.1b-chat-v1.0" (6 x Q8_0).
const std::vector<NamedPolicy>& named_policies();

struct PolicySelection {
  std::string name;
  std::vector<QuantOverride> overrides;
  std::string note;  // how it was chosen, printed by the CLI
};
// `policy` = "auto" (the named policy whose architecture guard matches cfg, else "uniform"),
// "uniform", or a named policy (throws if unknown or if its guard rejects cfg).
PolicySelection select_policy(std::string_view policy, const ModelConfig& cfg);

// Squared-error statistics of one tensor's reconstruction.
struct QuantError {
  double sse = 0.0, ref_ss = 0.0, max_abs_err = 0.0;
  double rel_rmse() const noexcept { return ref_ss > 0.0 ? std::sqrt(sse / ref_ss) : 0.0; }
};

// The storage type a tensor gets under `opt` (norms: F32; overrides; block types fall back to
// BF16 when a row length is not a multiple of 32). Same rules the .efq reader re-checks.
QType choose_qtype(const std::string& name, std::uint64_t cols, const QuantizeOptions& opt);

// Quantizes rows [row_begin, row_end) of a source tensor into `out` (packed rows of the
// target type). Deterministic: the bytes never depend on threading. Optionally measures error.
void quantize_tensor_rows(const SafetensorsFile& st, const TensorInfo& t, QType type, bool search,
                          std::uint64_t row_begin, std::uint64_t row_end, std::uint8_t* out,
                          std::vector<double>* row_sse = nullptr, std::vector<double>* row_ref = nullptr,
                          std::vector<double>* row_max = nullptr);

struct QuantizeResult {
  std::vector<QTensor> tensors;
  std::uint64_t file_bytes = 0, source_tensor_bytes = 0, bytes_per_token = 0, source_bytes_per_token = 0;
  // Tensor bytes the overrides add (or save) versus the uniform policy without them.
  std::int64_t override_extra_bytes = 0;
  double seconds = 0.0;
  std::vector<std::string> notes;
};

QuantizeResult quantize_model(const std::filesystem::path& model_dir, const std::filesystem::path& out_file,
                              const QuantizeOptions& opt,
                              const std::function<void(const QTensor&, std::size_t, std::size_t)>& progress = {});

}  // namespace ef
