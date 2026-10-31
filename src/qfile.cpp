// EdgeForge - .efq reader with full validation.
#include "ef/qfile.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <limits>
#include <numeric>

#include "ef/error.h"
#include "ef/format.h"

namespace ef {

namespace {

std::uint64_t read_u64(const std::uint8_t* p) {
  std::uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
  return v;
}
std::uint32_t read_u32(const std::uint8_t* p) {
  std::uint32_t v = 0;
  for (int i = 3; i >= 0; --i) v = (v << 8) | p[i];
  return v;
}

std::string lower(std::string s) {
  for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return s;
}

bool is_norm(const std::string& name) { return name.find("norm") != std::string::npos; }

QType parse_type_field(const json::Value& v, const char* what) {
  const auto t = qtype_from_string(v.as_string());
  if (!t) throw Error(std::string(".efq header: unknown ") + what + " type " + v.as_string());
  return *t;
}

}  // namespace

// ------------------------------------------------------------------ policy

bool glob_match(std::string_view p, std::string_view s) noexcept {
  std::size_t pi = 0, si = 0, star = std::string_view::npos, mark = 0;
  while (si < s.size()) {
    if (pi < p.size() && p[pi] == '*') {
      star = pi++;
      mark = si;
    } else if (pi < p.size() && p[pi] == s[si]) {
      ++pi;
      ++si;
    } else if (star != std::string_view::npos) {
      pi = star + 1;
      si = ++mark;
    } else {
      return false;
    }
  }
  while (pi < p.size() && p[pi] == '*') ++pi;
  return pi == p.size();
}

PolicyChoice policy_choice(const QuantPolicy& p, const std::string& name, std::uint64_t cols, bool use_overrides) {
  if (is_norm(name)) return {QType::F32, "norm"};
  PolicyChoice c{p.layers, "layers"};
  if (name == "model.embed_tokens.weight") c = {p.embed, "embed"};
  if (name == "lm_head.weight") c = {p.lm_head, "lm_head"};
  if (use_overrides) {
    for (const auto& o : p.overrides) {
      if (glob_match(o.pattern, name)) c = {o.type, "override"};
    }
  }
  if (qtype_is_block(c.type) && cols % kQK != 0) c = {QType::BF16, "fallback-bf16"};
  return c;
}

void check_overrides(const QuantPolicy& p, const std::vector<ExpectedTensor>& tensors) {
  for (const auto& o : p.overrides) {
    if (o.pattern.empty()) throw Error("override with an empty tensor pattern");
    std::size_t hits = 0;
    for (const auto& t : tensors) {
      if (!glob_match(o.pattern, t.name)) continue;
      if (is_norm(t.name)) {
        throw Error("override '" + o.pattern + "' matches norm tensor '" + t.name + "' (norms are always F32)");
      }
      ++hits;
    }
    if (hits == 0) throw Error("override '" + o.pattern + "' matches no tensor of this model");
  }
}

std::vector<QuantOverride> parse_overrides(std::string_view spec) {
  std::vector<QuantOverride> out;
  if (lower(std::string(spec)) == "none") return out;
  std::size_t pos = 0;
  while (pos <= spec.size()) {
    const std::size_t comma = std::min(spec.find(',', pos), spec.size());
    const std::string_view item = spec.substr(pos, comma - pos);
    const std::size_t eq = item.rfind('=');
    if (item.empty() || eq == std::string_view::npos || eq == 0 || eq + 1 == item.size()) {
      throw Error("override '" + std::string(item) + "' must look like TENSOR=TYPE (e.g. " +
                  "model.layers.0.self_attn.v_proj.weight=q8_0), or be 'none'");
    }
    const auto t = qtype_from_string(lower(std::string(item.substr(eq + 1))));
    if (!t) {
      throw Error("override '" + std::string(item) + "': type must be one of q4_0, q8_0, bf16, f16, f32");
    }
    out.push_back({std::string(item.substr(0, eq)), *t});
    pos = comma + 1;
  }
  return out;
}

std::string overrides_to_arg(const std::vector<QuantOverride>& v) {
  if (v.empty()) return "none";
  std::string s;
  for (const auto& o : v) s += (s.empty() ? "" : ",") + o.pattern + "=" + lower(qtype_name(o.type));
  return s;
}

// ------------------------------------------------------------------ reader

EfqLayout parse_efq(const std::uint8_t* file, std::uint64_t file_size) {
  if (file_size < 16) throw Error("file too small to be an .efq model (" + std::to_string(file_size) + " bytes)");
  if (std::memcmp(file, kEfqMagic, 4) != 0) throw Error("not an EdgeForge .efq file (bad magic)");
  const std::uint32_t version = read_u32(file + 4);
  if (version != kEfqVersion) {
    throw Error("unsupported .efq version " + std::to_string(version) + " (this build reads " +
                std::to_string(kEfqVersion) + ")");
  }
  EfqLayout L;
  L.header_len = read_u64(file + 8);
  if (L.header_len == 0 || L.header_len > 100'000'000 || L.header_len > file_size - 16) {
    throw Error("invalid .efq header length " + std::to_string(L.header_len));
  }
  const std::string_view text(reinterpret_cast<const char*>(file + 16), static_cast<std::size_t>(L.header_len));
  try {
    L.header = json::parse(text);
  } catch (const Error& e) {
    throw Error(std::string("invalid .efq header: ") + e.what());
  }
  if (L.header.at("format").as_string() != "edgeforge-quant") throw Error(".efq header: wrong \"format\"");
  L.config = parse_model_config(L.header.at("config"));
  const std::uint64_t data_start = 16 + L.header_len;

  for (const auto& e : L.header.at("tensors").items()) {
    QTensor t;
    t.name = e.at("name").as_string();
    const auto type = qtype_from_string(e.at("type").as_string());
    if (!type) throw Error("tensor '" + t.name + "': unknown type " + e.at("type").as_string());
    t.type = *type;
    if (const auto* r = e.find("rule")) t.rule = r->as_string();
    for (const auto& d : e.at("shape").items()) t.shape.push_back(d.as_uint64());
    if (t.shape.empty() || t.shape.size() > 2) throw Error("tensor '" + t.name + "': shape must be 1-D or 2-D");
    t.offset = e.at("offset").as_uint64();
    t.bytes = e.at("bytes").as_uint64();
    if (const auto* r = e.find("rel_rmse")) t.rel_rmse = r->as_double();
    if (const auto* r = e.find("max_abs_err")) t.max_abs_err = r->as_double();
    const std::uint64_t row = qtype_row_bytes(t.type, t.cols());
    if (t.rows() != 0 && row > std::numeric_limits<std::uint64_t>::max() / t.rows()) {
      throw Error("tensor '" + t.name + "': size overflows");
    }
    if (row * t.rows() != t.bytes) {
      throw Error("tensor '" + t.name + "': " + qtype_name(t.type) + " " + shape_to_string(t.shape) + " needs " +
                  std::to_string(row * t.rows()) + " bytes, header says " + std::to_string(t.bytes));
    }
    if (t.offset % kEfqAlign != 0) throw Error("tensor '" + t.name + "': offset is not 64-byte aligned");
    if (t.offset < data_start) throw Error("tensor '" + t.name + "': data overlaps the header");
    if (t.offset > file_size || t.bytes > file_size - t.offset) {
      throw Error("tensor '" + t.name + "': data runs past the end of the file (truncated?)");
    }
    L.tensors.push_back(std::move(t));
  }
  std::vector<std::size_t> order(L.tensors.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return L.tensors[a].offset < L.tensors[b].offset; });
  for (std::size_t i = 1; i < order.size(); ++i) {
    const QTensor& p = L.tensors[order[i - 1]];
    const QTensor& c = L.tensors[order[i]];
    if (c.offset < p.offset + p.bytes) throw Error("tensors '" + p.name + "' and '" + c.name + "' overlap");
  }

  // Exactly the tensors the config requires, with matching shapes.
  std::unordered_map<std::string, const QTensor*> by_name;
  for (const auto& t : L.tensors) {
    if (!by_name.emplace(t.name, &t).second) throw Error("duplicate tensor '" + t.name + "'");
  }
  const auto expected = expected_tensors(L.config);
  for (const auto& e : expected) {
    const auto it = by_name.find(e.name);
    if (it == by_name.end()) throw Error("missing tensor '" + e.name + "'");
    if (it->second->shape != e.shape) {
      throw Error("tensor '" + e.name + "' has shape " + shape_to_string(it->second->shape) + ", config needs " +
                  shape_to_string(e.shape));
    }
  }
  if (by_name.size() != expected.size()) throw Error(".efq file has tensors the config does not use");

  // The precision policy must explain every tensor's type (mixed precision is never silent).
  const json::Value& qs = L.header.at("quant");
  const json::Value* ov = qs.find("overrides");
  if (ov == nullptr) {
    throw Error(".efq header has no \"overrides\" table: written by an older pre-release build, re-run "
                "edgeforge quantize");
  }
  L.policy.layers = parse_type_field(qs.at("layers"), "layers");
  L.policy.lm_head = parse_type_field(qs.at("lm_head"), "lm_head");
  L.policy.embed = parse_type_field(qs.at("embed"), "embed");
  if (qs.at("norms").as_string() != "F32") throw Error(".efq header: norms must be F32");
  if (const auto* n = qs.find("policy")) L.policy.name = n->as_string();
  for (const auto& o : ov->items()) L.policy.overrides.push_back({o.at("pattern").as_string(), parse_type_field(o.at("type"), "override")});
  check_overrides(L.policy, expected);
  for (const auto& t : L.tensors) {
    const PolicyChoice c = policy_choice(L.policy, t.name, t.cols());
    if (t.type != c.type || t.rule != c.rule) {
      throw Error("tensor '" + t.name + "' is stored as " + qtype_name(t.type) + " (rule \"" + t.rule +
                  "\") but the file's quantization policy gives " + qtype_name(c.type) + " (rule \"" + c.rule + "\")");
    }
  }
  return L;
}

QuantFile QuantFile::open(const std::filesystem::path& path) {
  QuantFile q;
  q.file_ = MappedFile(path);
  try {
    EfqLayout L = parse_efq(q.file_.data(), q.file_.size());
    q.header_ = std::move(L.header);
    q.cfg_ = std::move(L.config);
    q.policy_ = std::move(L.policy);
    q.tensors_ = std::move(L.tensors);
  } catch (const Error& e) {
    throw Error("'" + path.string() + "': " + e.what());
  }
  for (std::size_t i = 0; i < q.tensors_.size(); ++i) {
    q.index_.emplace(q.tensors_[i].name, i);
    if (q.tensors_[i].name != "model.embed_tokens.weight") q.bytes_per_token_ += q.tensors_[i].bytes;
  }
  if (q.cfg_.tie_word_embeddings) q.bytes_per_token_ += q.at("model.embed_tokens.weight").bytes;
  return q;
}

const QTensor* QuantFile::find(const std::string& name) const {
  const auto it = index_.find(name);
  return it == index_.end() ? nullptr : &tensors_[it->second];
}

const QTensor& QuantFile::at(const std::string& name) const {
  if (const QTensor* t = find(name)) return *t;
  throw Error("tensor '" + name + "' not found in '" + path().string() + "'");
}

const std::uint8_t* QuantFile::row_data(const QTensor& t, std::uint64_t row) const {
  if (row >= t.rows()) throw Error("tensor '" + t.name + "': row " + std::to_string(row) + " out of range");
  return data(t) + row * qtype_row_bytes(t.type, t.cols());
}

void QuantFile::dequantize_row(const QTensor& t, std::uint64_t row, float* out) const {
  ef::dequantize_row(t.type, row_data(t, row), static_cast<std::size_t>(t.cols()), out);
}

}  // namespace ef
