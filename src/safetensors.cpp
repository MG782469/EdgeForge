// EdgeForge — safetensors reader implementation.
#include "ef/safetensors.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <numeric>

#include "ef/error.h"
#include "ef/fp16.h"
#include "ef/json.h"

static_assert(std::endian::native == std::endian::little,
              "safetensors stores little-endian data; big-endian hosts are not supported");

namespace ef {

namespace {

struct DTypeEntry {
  DType dtype;
  const char* name;
  std::size_t size;
};

constexpr DTypeEntry kDTypes[] = {
    {DType::BOOL, "BOOL", 1}, {DType::U8, "U8", 1},         {DType::I8, "I8", 1},
    {DType::F8_E4M3, "F8_E4M3", 1}, {DType::F8_E5M2, "F8_E5M2", 1}, {DType::U16, "U16", 2},
    {DType::I16, "I16", 2},   {DType::F16, "F16", 2},       {DType::BF16, "BF16", 2},
    {DType::U32, "U32", 4},   {DType::I32, "I32", 4},       {DType::F32, "F32", 4},
    {DType::U64, "U64", 8},   {DType::I64, "I64", 8},       {DType::F64, "F64", 8},
};

bool mul_overflows(std::uint64_t a, std::uint64_t b, std::uint64_t& out) {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) return true;
  out = a * b;
  return false;
}

std::uint64_t read_le_u64(const std::uint8_t* p) {
  std::uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
  return v;
}

std::string with_commas(std::uint64_t v) {
  std::string s = std::to_string(v);
  for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) s.insert(static_cast<std::size_t>(i), ",");
  return s;
}

TensorInfo parse_tensor_entry(const std::string& name, const json::Value& v) {
  const std::string where = "tensor '" + name + "': ";
  if (!v.is_object()) throw Error(where + "entry must be a JSON object");
  for (const auto& key : v.keys()) {
    if (key != "dtype" && key != "shape" && key != "data_offsets") {
      throw Error(where + "unexpected field \"" + key + "\"");
    }
  }
  const json::Value* dt = v.find("dtype");
  const json::Value* sh = v.find("shape");
  const json::Value* off = v.find("data_offsets");
  if (dt == nullptr) throw Error(where + "missing \"dtype\"");
  if (sh == nullptr) throw Error(where + "missing \"shape\"");
  if (off == nullptr) throw Error(where + "missing \"data_offsets\"");
  if (!dt->is_string()) throw Error(where + "\"dtype\" must be a string");
  const auto dtype = dtype_from_string(dt->as_string());
  if (!dtype) throw Error(where + "unsupported dtype \"" + dt->as_string() + "\"");

  TensorInfo t;
  t.name = name;
  t.dtype = *dtype;

  if (!sh->is_array()) throw Error(where + "\"shape\" must be an array");
  std::uint64_t numel = 1;
  for (const auto& d : sh->items()) {
    if (!d.is_integer() || d.as_int64() < 0) {
      throw Error(where + "shape entries must be non-negative integers");
    }
    const std::uint64_t dim = d.as_uint64();
    t.shape.push_back(dim);
    if (mul_overflows(numel, dim, numel)) throw Error(where + "shape element count overflows 64 bits");
  }
  t.numel = numel;

  if (!off->is_array() || off->size() != 2) {
    throw Error(where + "\"data_offsets\" must be an array of exactly 2 integers");
  }
  for (const auto& o : off->items()) {
    if (!o.is_integer() || o.as_int64() < 0) {
      throw Error(where + "\"data_offsets\" entries must be non-negative integers");
    }
  }
  t.begin = (*off)[0].as_uint64();
  t.end = (*off)[1].as_uint64();
  if (t.begin > t.end) {
    throw Error(where + "data_offsets begin (" + std::to_string(t.begin) + ") > end (" +
                std::to_string(t.end) + ")");
  }
  std::uint64_t expected_bytes = 0;
  if (mul_overflows(numel, dtype_size(t.dtype), expected_bytes)) {
    throw Error(where + "byte size overflows 64 bits");
  }
  if (expected_bytes != t.end - t.begin) {
    throw Error(where + "shape " + shape_to_string(t.shape) + " with dtype " + dtype_name(t.dtype) +
                " needs " + with_commas(expected_bytes) + " bytes, but data_offsets span " +
                with_commas(t.end - t.begin) + " bytes");
  }
  return t;
}

}  // namespace

// ---------------------------------------------------------------- dtypes

std::optional<DType> dtype_from_string(std::string_view s) noexcept {
  for (const auto& e : kDTypes) {
    if (s == e.name) return e.dtype;
  }
  return std::nullopt;
}

const char* dtype_name(DType d) noexcept {
  for (const auto& e : kDTypes) {
    if (e.dtype == d) return e.name;
  }
  return "?";
}

std::size_t dtype_size(DType d) noexcept {
  for (const auto& e : kDTypes) {
    if (e.dtype == d) return e.size;
  }
  return 0;
}

bool dtype_converts_to_f32(DType d) noexcept {
  return d == DType::F16 || d == DType::BF16 || d == DType::F32;
}

std::string shape_to_string(const std::vector<std::uint64_t>& shape) {
  std::string s = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i) s += ", ";
    s += std::to_string(shape[i]);
  }
  return s + "]";
}

// ---------------------------------------------------------------- header

SafetensorsHeader parse_safetensors_header(const std::uint8_t* file, std::uint64_t file_size) {
  if (file_size < 8) {
    throw Error("file is too small (" + std::to_string(file_size) +
                " bytes); a safetensors file starts with an 8-byte header length");
  }
  SafetensorsHeader h;
  h.header_len = read_le_u64(file);
  if (h.header_len == 0) throw Error("header length is 0");
  if (h.header_len > kMaxHeaderBytes) {
    throw Error("header length " + with_commas(h.header_len) + " exceeds the 100 MB limit (corrupt file?)");
  }
  if (h.header_len > file_size - 8) {
    throw Error("header length " + with_commas(h.header_len) + " exceeds the file size (" +
                with_commas(file_size) + " bytes; truncated file?)");
  }
  h.data_offset = 8 + h.header_len;
  h.data_size = file_size - h.data_offset;

  const std::string_view text(reinterpret_cast<const char*>(file + 8), static_cast<std::size_t>(h.header_len));
  if (text.front() != '{') throw Error("header does not start with '{'");

  json::Value root;
  try {
    root = json::parse(text);
  } catch (const Error& e) {
    throw Error(std::string("invalid header: ") + e.what());
  }
  if (!root.is_object()) throw Error("header must be a JSON object");

  const auto& keys = root.keys();
  const auto& vals = root.items();
  for (std::size_t i = 0; i < keys.size(); ++i) {
    if (keys[i] == "__metadata__") {
      if (!vals[i].is_object()) throw Error("\"__metadata__\" must be a JSON object");
      for (std::size_t k = 0; k < vals[i].keys().size(); ++k) {
        const auto& mv = vals[i].items()[k];
        if (!mv.is_string()) {
          throw Error("\"__metadata__\" value for \"" + vals[i].keys()[k] + "\" must be a string");
        }
        h.metadata[vals[i].keys()[k]] = mv.as_string();
      }
      continue;
    }
    TensorInfo t = parse_tensor_entry(keys[i], vals[i]);
    if (t.end > h.data_size) {
      throw Error("tensor '" + t.name + "': data_offsets end " + with_commas(t.end) +
                  " exceeds the data region size " + with_commas(h.data_size) + " (truncated file?)");
    }
    h.index.emplace(t.name, h.tensors.size());
    h.tensors.push_back(std::move(t));
  }

  // The tensors must tile [0, data_size) exactly: no overlap, no gap, no trailing bytes.
  std::vector<std::size_t> order(h.tensors.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    const auto& ta = h.tensors[a];
    const auto& tb = h.tensors[b];
    return ta.begin != tb.begin ? ta.begin < tb.begin : ta.end < tb.end;
  });
  std::uint64_t cursor = 0;
  const TensorInfo* prev = nullptr;
  for (std::size_t idx : order) {
    const TensorInfo& t = h.tensors[idx];
    if (t.begin < cursor) {
      throw Error("tensors '" + (prev ? prev->name : std::string("?")) + "' and '" + t.name +
                  "' overlap in the data region");
    }
    if (t.begin > cursor) {
      throw Error("gap of " + with_commas(t.begin - cursor) + " bytes in the data region before tensor '" +
                  t.name + "' (offset " + with_commas(cursor) + ")");
    }
    cursor = t.end;
    prev = &t;
  }
  if (cursor != h.data_size) {
    throw Error("data region has " + with_commas(h.data_size) + " bytes but tensors cover only " +
                with_commas(cursor) + " (unexpected trailing bytes)");
  }
  return h;
}

// ---------------------------------------------------------------- conversion

void convert_to_f32(DType dtype, const std::uint8_t* src, std::uint64_t count, float* out) {
  switch (dtype) {
    case DType::F32:
      std::memcpy(out, src, static_cast<std::size_t>(count) * 4);
      return;
    case DType::BF16:
      for (std::uint64_t i = 0; i < count; ++i) {
        std::uint16_t h;
        std::memcpy(&h, src + 2 * i, 2);
        out[i] = bf16_to_f32(h);
      }
      return;
    case DType::F16:
      for (std::uint64_t i = 0; i < count; ++i) {
        std::uint16_t h;
        std::memcpy(&h, src + 2 * i, 2);
        out[i] = fp16_to_f32(h);
      }
      return;
    default:
      throw Error(std::string("dtype ") + dtype_name(dtype) + " cannot be converted to f32");
  }
}

// ---------------------------------------------------------------- file

SafetensorsFile SafetensorsFile::open(const std::filesystem::path& path) {
  SafetensorsFile f;
  f.file_ = MappedFile(path);
  try {
    f.header_ = parse_safetensors_header(f.file_.data(), f.file_.size());
  } catch (const Error& e) {
    throw Error("safetensors '" + path.string() + "': " + e.what());
  }
  return f;
}

const TensorInfo* SafetensorsFile::find(std::string_view name) const {
  const auto it = header_.index.find(std::string(name));
  return it == header_.index.end() ? nullptr : &header_.tensors[it->second];
}

const TensorInfo& SafetensorsFile::at(std::string_view name) const {
  if (const TensorInfo* t = find(name)) return *t;
  // Suggest names that share the longest prefix — usually catches typos.
  std::size_t best = 0;
  std::vector<std::string> hints;
  for (const auto& t : header_.tensors) {
    std::size_t p = 0;
    while (p < t.name.size() && p < name.size() && t.name[p] == name[p]) ++p;
    if (p > best) {
      best = p;
      hints.clear();
    }
    if (p == best && p > 0 && hints.size() < 3) hints.push_back(t.name);
  }
  std::string msg = "tensor '" + std::string(name) + "' not found in '" + path().string() + "' (" +
                    std::to_string(header_.tensors.size()) + " tensors)";
  if (!hints.empty()) {
    msg += "; closest: ";
    for (std::size_t i = 0; i < hints.size(); ++i) msg += (i ? ", " : "") + hints[i];
  }
  throw Error(msg);
}

std::span<const std::uint8_t> SafetensorsFile::bytes(const TensorInfo& t) const {
  return {file_.data() + header_.data_offset + t.begin, static_cast<std::size_t>(t.nbytes())};
}

void SafetensorsFile::read_f32(const TensorInfo& t, std::uint64_t first, std::uint64_t count,
                               float* out) const {
  if (!dtype_converts_to_f32(t.dtype)) {
    throw Error("tensor '" + t.name + "': dtype " + dtype_name(t.dtype) + " cannot be read as f32");
  }
  if (first > t.numel || count > t.numel - first) {
    throw Error("tensor '" + t.name + "': element range [" + std::to_string(first) + ", " +
                std::to_string(first + count) + ") is outside [0, " + std::to_string(t.numel) + ")");
  }
  const std::uint8_t* base = bytes(t).data() + first * dtype_size(t.dtype);
  convert_to_f32(t.dtype, base, count, out);
}

float SafetensorsFile::value_f32(const TensorInfo& t, std::uint64_t index) const {
  float v = 0.0f;
  read_f32(t, index, 1, &v);
  return v;
}

std::uint32_t SafetensorsFile::raw_bits(const TensorInfo& t, std::uint64_t index) const {
  if (index >= t.numel) {
    throw Error("tensor '" + t.name + "': index " + std::to_string(index) + " out of range");
  }
  const std::size_t sz = dtype_size(t.dtype);
  const std::uint8_t* p = bytes(t).data() + index * sz;
  if (sz == 2) {
    std::uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
  }
  if (sz == 4) {
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
  }
  throw Error("tensor '" + t.name + "': raw_bits supports 16/32-bit dtypes only");
}

}  // namespace ef
