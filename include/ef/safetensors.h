// EdgeForge — safetensors reader.
//
// File layout: [u64 little-endian N][N bytes JSON header][tensor data region].
// Header: { "<name>": {"dtype": "BF16", "shape": [..], "data_offsets": [b, e]},
//           "__metadata__": {"k": "v", ...} }   (offsets are relative to the data region)
//
// Validation follows the reference implementation: header length bounds,
// strict JSON (no duplicate names), known dtypes, shape*dtype == byte span,
// offsets inside the file, and the tensors must tile the data region exactly
// (no overlaps, no holes, no trailing bytes). Tensor data is never assumed to
// be aligned — values are read with memcpy.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ef/mapped_file.h"

namespace ef {

enum class DType : std::uint8_t {
  BOOL, U8, I8, F8_E4M3, F8_E5M2, U16, I16, F16, BF16, U32, I32, F32, U64, I64, F64
};

std::optional<DType> dtype_from_string(std::string_view s) noexcept;
const char* dtype_name(DType d) noexcept;
std::size_t dtype_size(DType d) noexcept;
bool dtype_converts_to_f32(DType d) noexcept;  // F16, BF16, F32

struct TensorInfo {
  std::string name;
  DType dtype = DType::F32;
  std::vector<std::uint64_t> shape;
  std::uint64_t numel = 0;
  std::uint64_t begin = 0;  // byte offset relative to the data region
  std::uint64_t end = 0;    // one past the last byte, relative to the data region
  std::uint64_t nbytes() const noexcept { return end - begin; }
};

std::string shape_to_string(const std::vector<std::uint64_t>& shape);

struct SafetensorsHeader {
  std::uint64_t header_len = 0;   // N
  std::uint64_t data_offset = 0;  // 8 + N (absolute file offset of the data region)
  std::uint64_t data_size = 0;    // file size - 8 - N
  std::vector<TensorInfo> tensors;                       // in header order
  std::unordered_map<std::string, std::size_t> index;    // name -> position in tensors
  std::map<std::string, std::string> metadata;           // "__metadata__"
};

// Hard cap from the safetensors spec; protects against corrupt length fields.
inline constexpr std::uint64_t kMaxHeaderBytes = 100'000'000;

// Parses and fully validates the header of a safetensors file held in memory
// (only the first 8+N bytes are read; file_size is needed for bounds checks).
SafetensorsHeader parse_safetensors_header(const std::uint8_t* file, std::uint64_t file_size);

// Converts `count` elements of a float tensor (F16/BF16/F32) to f32.
void convert_to_f32(DType dtype, const std::uint8_t* src, std::uint64_t count, float* out);

class SafetensorsFile {
 public:
  static SafetensorsFile open(const std::filesystem::path& path);  // throws ef::Error

  const std::filesystem::path& path() const noexcept { return file_.path(); }
  std::uint64_t file_size() const noexcept { return file_.size(); }
  const SafetensorsHeader& header() const noexcept { return header_; }
  const std::vector<TensorInfo>& tensors() const noexcept { return header_.tensors; }

  const TensorInfo* find(std::string_view name) const;
  const TensorInfo& at(std::string_view name) const;  // throws with a helpful message

  // Raw bytes of one tensor (a view into the mapping; nothing is copied).
  std::span<const std::uint8_t> bytes(const TensorInfo& t) const;

  // Reads elements [first, first+count) of a float tensor as f32. Bounds-checked.
  void read_f32(const TensorInfo& t, std::uint64_t first, std::uint64_t count, float* out) const;
  float value_f32(const TensorInfo& t, std::uint64_t index) const;

  // Raw bit pattern of element `index` (16- or 32-bit dtypes), for exact comparisons.
  std::uint32_t raw_bits(const TensorInfo& t, std::uint64_t index) const;

 private:
  MappedFile file_;
  SafetensorsHeader header_;
};

}  // namespace ef
