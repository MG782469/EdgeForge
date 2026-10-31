// EdgeForge — config.json parsing and checkpoint checks.
#include "ef/model_config.h"

#include <cmath>
#include <fstream>
#include <iterator>
#include <sstream>
#include <unordered_map>

#include "ef/error.h"

namespace ef {

namespace {

const json::Value* get(const json::Value& root, const char* key) { return root.find(key); }

std::int64_t req_int(const json::Value& root, const char* key, std::int64_t lo, std::int64_t hi) {
  const json::Value* v = get(root, key);
  if (v == nullptr) throw Error(std::string("config.json: missing required field '") + key + "'");
  if (!v->is_integer()) {
    throw Error(std::string("config.json: '") + key + "' must be an integer, got " +
                (v->is_number() ? v->number_text() : json::type_name(v->type())));
  }
  const std::int64_t x = v->as_int64();
  if (x < lo || x > hi) {
    throw Error(std::string("config.json: '") + key + "' = " + std::to_string(x) + " is outside [" +
                std::to_string(lo) + ", " + std::to_string(hi) + "]");
  }
  return x;
}

std::int64_t opt_int(const json::Value& root, const char* key, std::int64_t def, std::int64_t lo,
                     std::int64_t hi) {
  const json::Value* v = get(root, key);
  if (v == nullptr || v->is_null()) return def;
  return req_int(root, key, lo, hi);
}

double req_double(const json::Value& root, const char* key) {
  const json::Value* v = get(root, key);
  if (v == nullptr) throw Error(std::string("config.json: missing required field '") + key + "'");
  if (!v->is_number()) {
    throw Error(std::string("config.json: '") + key + "' must be a number, got " + json::type_name(v->type()));
  }
  return v->as_double();
}

double opt_double(const json::Value& root, const char* key, double def) {
  const json::Value* v = get(root, key);
  if (v == nullptr || v->is_null()) return def;
  return req_double(root, key);
}

bool opt_bool(const json::Value& root, const char* key, bool def) {
  const json::Value* v = get(root, key);
  if (v == nullptr || v->is_null()) return def;
  if (!v->is_bool()) throw Error(std::string("config.json: '") + key + "' must be true or false");
  return v->as_bool();
}

std::string opt_string(const json::Value& root, const char* key, const std::string& def) {
  const json::Value* v = get(root, key);
  if (v == nullptr || v->is_null()) return def;
  if (!v->is_string()) throw Error(std::string("config.json: '") + key + "' must be a string");
  return v->as_string();
}

constexpr std::int64_t kMaxDim = 1 << 20;

}  // namespace

ModelConfig parse_model_config(const json::Value& root) {
  if (!root.is_object()) throw Error("config.json: top level must be a JSON object");
  ModelConfig c;

  c.model_type = opt_string(root, "model_type", "");
  if (c.model_type != "llama") {
    throw Error("config.json: model_type '" + c.model_type + "' is not supported (EdgeForge targets 'llama')");
  }
  if (const json::Value* a = get(root, "architectures"); a != nullptr && a->is_array()) {
    for (const auto& e : a->items()) {
      if (e.is_string()) c.architectures.push_back(e.as_string());
    }
  }
  c.hidden_act = opt_string(root, "hidden_act", "silu");
  if (c.hidden_act != "silu") {
    throw Error("config.json: hidden_act '" + c.hidden_act + "' is not supported (expected 'silu')");
  }
  c.torch_dtype = opt_string(root, "torch_dtype", opt_string(root, "dtype", ""));

  c.hidden_size = req_int(root, "hidden_size", 1, kMaxDim);
  c.intermediate_size = req_int(root, "intermediate_size", 1, kMaxDim);
  c.num_hidden_layers = req_int(root, "num_hidden_layers", 1, 4096);
  c.num_attention_heads = req_int(root, "num_attention_heads", 1, 4096);
  c.num_key_value_heads = opt_int(root, "num_key_value_heads", c.num_attention_heads, 1, 4096);
  c.vocab_size = req_int(root, "vocab_size", 1, 1 << 24);
  c.max_position_embeddings = opt_int(root, "max_position_embeddings", 2048, 1, 1 << 24);
  c.rms_norm_eps = req_double(root, "rms_norm_eps");
  c.rope_theta = opt_double(root, "rope_theta", 10000.0);
  c.tie_word_embeddings = opt_bool(root, "tie_word_embeddings", false);
  c.attention_bias = opt_bool(root, "attention_bias", false);
  c.mlp_bias = opt_bool(root, "mlp_bias", false);
  c.bos_token_id = opt_int(root, "bos_token_id", -1, -1, 1 << 24);
  c.eos_token_id = opt_int(root, "eos_token_id", -1, -1, 1 << 24);

  if (c.hidden_size % c.num_attention_heads != 0) {
    throw Error("config.json: hidden_size " + std::to_string(c.hidden_size) +
                " is not divisible by num_attention_heads " + std::to_string(c.num_attention_heads));
  }
  c.head_dim = opt_int(root, "head_dim", c.hidden_size / c.num_attention_heads, 1, kMaxDim);
  if (c.head_dim * c.num_attention_heads != c.hidden_size) {
    throw Error("config.json: head_dim * num_attention_heads must equal hidden_size");
  }
  if (c.head_dim % 2 != 0) throw Error("config.json: head_dim must be even for RoPE");
  if (c.num_attention_heads % c.num_key_value_heads != 0) {
    throw Error("config.json: num_attention_heads " + std::to_string(c.num_attention_heads) +
                " is not divisible by num_key_value_heads " + std::to_string(c.num_key_value_heads));
  }
  if (!(c.rms_norm_eps > 0.0) || !std::isfinite(c.rms_norm_eps)) {
    throw Error("config.json: rms_norm_eps must be a positive finite number");
  }
  if (!(c.rope_theta > 0.0) || !std::isfinite(c.rope_theta)) {
    throw Error("config.json: rope_theta must be a positive finite number");
  }
  if (const json::Value* rs = get(root, "rope_scaling"); rs != nullptr && !rs->is_null()) {
    throw Error("config.json: rope_scaling is set; scaled RoPE is not supported yet");
  }
  if (c.attention_bias) throw Error("config.json: attention_bias=true is not supported");
  if (c.mlp_bias) throw Error("config.json: mlp_bias=true is not supported");
  return c;
}

ModelConfig load_model_config(const std::filesystem::path& config_json) {
  std::ifstream in(config_json, std::ios::binary);
  if (!in) throw Error("cannot open '" + config_json.string() + "'");
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  try {
    return parse_model_config(json::parse(text));
  } catch (const Error& e) {
    throw Error("'" + config_json.string() + "': " + e.what());
  }
}

std::vector<ExpectedTensor> expected_tensors(const ModelConfig& c) {
  const auto D = static_cast<std::uint64_t>(c.hidden_size);
  const auto F = static_cast<std::uint64_t>(c.intermediate_size);
  const auto V = static_cast<std::uint64_t>(c.vocab_size);
  const auto KV = static_cast<std::uint64_t>(c.kv_dim());
  std::vector<ExpectedTensor> out;
  out.push_back({"model.embed_tokens.weight", {V, D}});
  for (std::int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    out.push_back({p + "self_attn.q_proj.weight", {D, D}});
    out.push_back({p + "self_attn.k_proj.weight", {KV, D}});
    out.push_back({p + "self_attn.v_proj.weight", {KV, D}});
    out.push_back({p + "self_attn.o_proj.weight", {D, D}});
    out.push_back({p + "mlp.gate_proj.weight", {F, D}});
    out.push_back({p + "mlp.up_proj.weight", {F, D}});
    out.push_back({p + "mlp.down_proj.weight", {D, F}});
    out.push_back({p + "input_layernorm.weight", {D}});
    out.push_back({p + "post_attention_layernorm.weight", {D}});
  }
  out.push_back({"model.norm.weight", {D}});
  if (!c.tie_word_embeddings) out.push_back({"lm_head.weight", {V, D}});
  return out;
}

std::uint64_t expected_param_count(const ModelConfig& c) {
  std::uint64_t n = 0;
  for (const auto& t : expected_tensors(c)) {
    std::uint64_t k = 1;
    for (auto d : t.shape) k *= d;
    n += k;
  }
  return n;
}

CheckReport check_checkpoint(const ModelConfig& cfg, const std::vector<TensorInfo>& tensors) {
  CheckReport r;
  std::unordered_map<std::string, const TensorInfo*> by_name;
  for (const auto& t : tensors) by_name.emplace(t.name, &t);

  const auto expected = expected_tensors(cfg);
  std::unordered_map<std::string, bool> wanted;
  std::size_t missing = 0;
  for (const auto& e : expected) {
    wanted.emplace(e.name, true);
    const auto it = by_name.find(e.name);
    if (it == by_name.end()) {
      if (++missing <= 10) r.errors.push_back("missing tensor '" + e.name + "'");
      continue;
    }
    const TensorInfo& t = *it->second;
    if (t.shape != e.shape) {
      r.errors.push_back("tensor '" + e.name + "' has shape " + shape_to_string(t.shape) + ", expected " +
                         shape_to_string(e.shape));
      continue;
    }
    if (!dtype_converts_to_f32(t.dtype)) {
      r.errors.push_back("tensor '" + e.name + "' has non-float dtype " + dtype_name(t.dtype));
      continue;
    }
    ++r.matched_tensors;
    r.matched_params += t.numel;
  }
  if (missing > 10) r.errors.push_back("... and " + std::to_string(missing - 10) + " more missing tensors");

  for (const auto& t : tensors) {
    if (wanted.count(t.name)) continue;
    const std::string suffix = "rotary_emb.inv_freq";
    if (t.name.size() >= suffix.size() &&
        t.name.compare(t.name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      r.warnings.push_back("ignoring buffer '" + t.name + "' (RoPE frequencies are computed at runtime)");
    } else if (cfg.tie_word_embeddings && t.name == "lm_head.weight") {
      r.warnings.push_back("config ties embeddings but the checkpoint also has lm_head.weight");
    } else {
      r.errors.push_back("unexpected tensor '" + t.name + "'");
    }
  }
  return r;
}

}  // namespace ef
