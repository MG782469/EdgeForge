// EdgeForge - Stage 5: the .efq quantized model file.
//
//   bytes [0,4)    magic "EFQ1"
//   bytes [4,8)    u32 little-endian format version (1)
//   bytes [8,16)   u64 little-endian header length N
//   bytes [16,16+N) JSON header (space-padded so tensor data starts 64-byte aligned):
//       {"format":"edgeforge-quant","version":1,
//        "quant":{"layers","lm_head","embed","norms","block","scale",
//                 "policy":"<name>", "overrides":[{"pattern","type"}, ...]},
//        "source":{...}, "config":{...verbatim config.json...},
//        "tensors":[{"name","type","rule","shape","offset","bytes","rel_rmse","max_abs_err"}, ...]}
//   tensor data: each tensor at an absolute 64-byte-aligned offset, rows contiguous,
//       stored in forward-pass order (embed, layer 0 ... layer 21, norm, lm_head) so decoding
//       streams the file front to back.
//
// Mixed precision: every tensor records its own type and the policy rule that chose it
// ("layers", "lm_head", "embed", "norm", "override", "fallback-bf16"). The reader recomputes
// the choice from the "quant" policy and rejects a file where any tensor disagrees, so a
// per-tensor exception can never be silent.
//
// One file holds everything the runtime needs except the tokenizer.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ef/json.h"
#include "ef/mapped_file.h"
#include "ef/model_config.h"
#include "ef/quant.h"

namespace ef {

inline constexpr char kEfqMagic[4] = {'E', 'F', 'Q', '1'};
inline constexpr std::uint32_t kEfqVersion = 1;
inline constexpr std::uint64_t kEfqAlign = 64;

// ---- quantization policy -----------------------------------------------------------------

// A per-tensor override: every tensor whose name matches `pattern` (exact name, or a glob
// where '*' matches any run of characters) is stored as `type`. If several overrides match
// a tensor, the last one wins. Norm tensors cannot be overridden (always F32).
struct QuantOverride {
  std::string pattern;
  QType type = QType::Q8_0;
};

struct QuantPolicy {
  QType layers = QType::Q4_0;   // attention + MLP matrices
  QType lm_head = QType::Q8_0;  // output projection
  QType embed = QType::BF16;    // token embedding table
  std::vector<QuantOverride> overrides;
  std::string name;             // label only ("uniform", a named policy, "custom"); never changes types
};

struct PolicyChoice {
  QType type = QType::F32;
  std::string rule;  // layers | lm_head | embed | norm | override | fallback-bf16
};

bool glob_match(std::string_view pattern, std::string_view text) noexcept;

// Storage type of one tensor. Block types fall back to BF16 (lossless) when the row length is
// not a multiple of 32. use_overrides = false gives the uniform baseline the overrides replace.
PolicyChoice policy_choice(const QuantPolicy& p, const std::string& name, std::uint64_t cols,
                           bool use_overrides = true);

// Throws unless every override pattern matches at least one tensor and none matches a norm.
void check_overrides(const QuantPolicy& p, const std::vector<ExpectedTensor>& tensors);

// "NAME=TYPE,PATTERN=TYPE,..." or "none" (case-insensitive types: q4_0, q8_0, bf16, f16, f32).
std::vector<QuantOverride> parse_overrides(std::string_view spec);
// Inverse of parse_overrides ("none" when empty), lowercase types: ready for --override.
std::string overrides_to_arg(const std::vector<QuantOverride>& v);

// ---- file ---------------------------------------------------------------------------------

struct QTensor {
  std::string name;
  QType type = QType::F32;
  std::string rule;                  // policy rule that chose `type` (see PolicyChoice)
  std::vector<std::uint64_t> shape;  // [rows, cols] or [n]
  std::uint64_t offset = 0;          // absolute file offset
  std::uint64_t bytes = 0;
  double rel_rmse = 0.0;             // ||W - deq(Q(W))|| / ||W|| measured by the quantizer
  double max_abs_err = 0.0;
  std::uint64_t rows() const noexcept { return shape.size() == 2 ? shape[0] : 1; }
  std::uint64_t cols() const noexcept { return shape.empty() ? 0 : shape.back(); }
};

class QuantFile {
 public:
  static QuantFile open(const std::filesystem::path& path);  // validates everything, throws ef::Error

  const std::filesystem::path& path() const noexcept { return file_.path(); }
  std::uint64_t file_size() const noexcept { return file_.size(); }
  const json::Value& header() const noexcept { return header_; }
  const ModelConfig& config() const noexcept { return cfg_; }
  const QuantPolicy& policy() const noexcept { return policy_; }
  const std::vector<QTensor>& tensors() const noexcept { return tensors_; }
  const QTensor* find(const std::string& name) const;
  const QTensor& at(const std::string& name) const;
  const std::uint8_t* data(const QTensor& t) const noexcept { return file_.data() + t.offset; }
  const std::uint8_t* row_data(const QTensor& t, std::uint64_t row) const;
  void dequantize_row(const QTensor& t, std::uint64_t row, float* out) const;
  // Bytes streamed per decoded token: everything except the embedding table (one row).
  std::uint64_t bytes_per_token() const noexcept { return bytes_per_token_; }

 private:
  MappedFile file_;
  json::Value header_;
  ModelConfig cfg_;
  QuantPolicy policy_;
  std::vector<QTensor> tensors_;
  std::unordered_map<std::string, std::size_t> index_;
  std::uint64_t bytes_per_token_ = 0;
};

// Parses and validates a header + tensor table against a file of `file_size` bytes (pure
// function: used by QuantFile::open and by tests on in-memory buffers).
struct EfqLayout {
  std::uint64_t header_len = 0;
  json::Value header;
  ModelConfig config;
  QuantPolicy policy;
  std::vector<QTensor> tensors;
};
EfqLayout parse_efq(const std::uint8_t* file, std::uint64_t file_size);

}  // namespace ef
