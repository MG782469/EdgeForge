// EdgeForge - reader for golden reference data (tools/dump_reference.py output)
// and numeric comparison helpers. See docs/golden_format.md.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "ef/json.h"

namespace ef {

struct GoldenTensor {
  std::string key;
  std::vector<std::uint64_t> shape;
  std::vector<float> f32;         // filled for dtype "f32"
  std::vector<std::int32_t> i32;  // filled for dtype "i32"
  std::uint64_t numel() const noexcept;
  // Row r of a 2-D (or leading-dim) f32 tensor.
  const float* row(std::size_t r) const;
  std::size_t row_size() const;   // product of all dims except the first
};

class GoldenPrompt {
 public:
  // dir = tests/golden/prompts/<name>; reads and validates manifest.json.
  static GoldenPrompt open(const std::filesystem::path& dir);

  const std::filesystem::path& dir() const noexcept { return dir_; }
  const json::Value& manifest() const noexcept { return manifest_; }
  bool has(const std::string& key) const;
  // Loads one tensor; checks dtype and that the file size equals numel * 4.
  GoldenTensor load(const std::string& key) const;
  std::vector<std::int32_t> input_ids() const;

 private:
  std::filesystem::path dir_;
  json::Value manifest_;
};

struct DiffStats {
  std::size_t n = 0;
  double max_abs_err = 0.0;   // max |ours - ref|
  double max_abs_ref = 0.0;   // max |ref|
  double rel_err = 0.0;       // max_abs_err / max(max_abs_ref, 1e-30)
  double rms_err = 0.0;       // sqrt(mean((ours - ref)^2))
  double cosine = 1.0;        // cosine similarity of the two vectors
  std::size_t worst_index = 0;
  std::size_t non_finite = 0; // NaN/Inf in ours
};

DiffStats compare(const float* ours, const float* ref, std::size_t n);

}  // namespace ef
