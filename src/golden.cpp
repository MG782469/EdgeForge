// EdgeForge - golden reference reader and comparison helpers.
#include "ef/golden.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>

#include "ef/error.h"

namespace ef {

std::uint64_t GoldenTensor::numel() const noexcept {
  std::uint64_t n = 1;
  for (auto d : shape) n *= d;
  return n;
}

std::size_t GoldenTensor::row_size() const {
  if (shape.empty()) throw Error("golden '" + key + "': scalar has no rows");
  std::uint64_t n = 1;
  for (std::size_t i = 1; i < shape.size(); ++i) n *= shape[i];
  return static_cast<std::size_t>(n);
}

const float* GoldenTensor::row(std::size_t r) const {
  if (shape.empty() || r >= shape[0]) throw Error("golden '" + key + "': row " + std::to_string(r) + " out of range");
  if (f32.empty()) throw Error("golden '" + key + "': not an f32 tensor");
  return f32.data() + r * row_size();
}

GoldenPrompt GoldenPrompt::open(const std::filesystem::path& dir) {
  const auto path = dir / "manifest.json";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw Error("cannot open golden manifest '" + path.string() + "' (run tools/dump_reference.py)");
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  GoldenPrompt g;
  g.dir_ = dir;
  try {
    g.manifest_ = json::parse(text);
    if (!g.manifest_.at("files").is_object()) throw Error("\"files\" must be an object");
  } catch (const Error& e) {
    throw Error("'" + path.string() + "': " + e.what());
  }
  return g;
}

bool GoldenPrompt::has(const std::string& key) const { return manifest_.at("files").find(key) != nullptr; }

GoldenTensor GoldenPrompt::load(const std::string& key) const {
  const json::Value* e = manifest_.at("files").find(key);
  if (e == nullptr) throw Error("golden tensor '" + key + "' not listed in " + (dir_ / "manifest.json").string());
  GoldenTensor t;
  t.key = key;
  for (const auto& d : e->at("shape").items()) t.shape.push_back(d.as_uint64());
  const std::string dtype = e->at("dtype").as_string();
  if (dtype != "f32" && dtype != "i32") throw Error("golden '" + key + "': unsupported dtype " + dtype);
  const auto path = dir_ / e->at("file").as_string();
  std::ifstream in(path, std::ios::binary);
  if (!in) throw Error("cannot open golden file '" + path.string() + "'");
  const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const std::uint64_t expected = t.numel() * 4;
  if (bytes.size() != expected) {
    throw Error("golden '" + key + "': file has " + std::to_string(bytes.size()) + " bytes, shape needs " +
                std::to_string(expected));
  }
  if (dtype == "f32") {
    t.f32.resize(static_cast<std::size_t>(t.numel()));
    std::memcpy(t.f32.data(), bytes.data(), bytes.size());
  } else {
    t.i32.resize(static_cast<std::size_t>(t.numel()));
    std::memcpy(t.i32.data(), bytes.data(), bytes.size());
  }
  return t;
}

std::vector<std::int32_t> GoldenPrompt::input_ids() const { return load("input_ids").i32; }

DiffStats compare(const float* ours, const float* ref, std::size_t n) {
  DiffStats s;
  s.n = n;
  double dot = 0.0, na = 0.0, nb = 0.0, se = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double a = static_cast<double>(ours[i]);
    const double b = static_cast<double>(ref[i]);
    if (!std::isfinite(a)) {
      ++s.non_finite;
      continue;
    }
    const double e = std::fabs(a - b);
    if (e > s.max_abs_err) {
      s.max_abs_err = e;
      s.worst_index = i;
    }
    s.max_abs_ref = std::max(s.max_abs_ref, std::fabs(b));
    dot += a * b;
    na += a * a;
    nb += b * b;
    se += (a - b) * (a - b);
  }
  s.rel_err = s.max_abs_err / std::max(s.max_abs_ref, 1e-30);
  s.rms_err = n ? std::sqrt(se / static_cast<double>(n)) : 0.0;
  s.cosine = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : (na == nb ? 1.0 : 0.0);
  return s;
}

}  // namespace ef
