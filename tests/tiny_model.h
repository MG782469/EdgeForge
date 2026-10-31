// EdgeForge - a tiny random Llama checkpoint written to a temp folder, for tests.
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "ef/fp16.h"
#include "test_util.h"

namespace ef_test {

using namespace ef;
namespace fs = std::filesystem;

struct Tiny {
  std::size_t D = 16, H = 4, KV = 2, hd = 4, F = 24, V = 11, L = 2;
  double eps = 1e-5, theta = 10000.0;
  bool tied = false;
};

// Optional hook to reshape the random weights: (tensor name, flat index, random value) -> value.
using TinyTweak = std::function<float(const std::string&, std::uint64_t, float)>;

struct TinyDir {
  fs::path path;
  std::map<std::string, std::vector<double>> w;  // exact (BF16-rounded) weights as double
  explicit TinyDir(const Tiny& t, unsigned seed = 1, bool drop_one = false, const TinyTweak& tweak = {}) {
    static std::atomic<int> n{0};
    path = fs::temp_directory_path() / ("edgeforge_tiny_" + std::to_string(n++) + "_" +
                                        std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    fs::create_directories(path);
    std::ofstream(path / "config.json")
        << "{\"model_type\":\"llama\",\"hidden_size\":" << t.D << ",\"intermediate_size\":" << t.F
        << ",\"num_hidden_layers\":" << t.L << ",\"num_attention_heads\":" << t.H
        << ",\"num_key_value_heads\":" << t.KV << ",\"vocab_size\":" << t.V
        << ",\"rms_norm_eps\":1e-05,\"rope_theta\":10000.0,\"max_position_embeddings\":256,"
        << "\"tie_word_embeddings\":" << (t.tied ? "true" : "false") << ",\"bos_token_id\":1,\"eos_token_id\":2}";
    std::vector<std::pair<std::string, std::vector<std::uint64_t>>> list = {{"model.embed_tokens.weight", {t.V, t.D}}};
    for (std::size_t l = 0; l < t.L; ++l) {
      const std::string p = "model.layers." + std::to_string(l) + ".";
      list.push_back({p + "self_attn.q_proj.weight", {t.D, t.D}});
      list.push_back({p + "self_attn.k_proj.weight", {t.KV * t.hd, t.D}});
      list.push_back({p + "self_attn.v_proj.weight", {t.KV * t.hd, t.D}});
      list.push_back({p + "self_attn.o_proj.weight", {t.D, t.D}});
      list.push_back({p + "mlp.gate_proj.weight", {t.F, t.D}});
      list.push_back({p + "mlp.up_proj.weight", {t.F, t.D}});
      list.push_back({p + "mlp.down_proj.weight", {t.D, t.F}});
      list.push_back({p + "input_layernorm.weight", {t.D}});
      list.push_back({p + "post_attention_layernorm.weight", {t.D}});
    }
    list.push_back({"model.norm.weight", {t.D}});
    if (!t.tied) list.push_back({"lm_head.weight", {t.V, t.D}});
    if (drop_one) list.erase(list.begin() + 3);

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::string header = "{";
    std::vector<std::uint8_t> data;
    for (const auto& [name, shape] : list) {
      std::uint64_t n_el = 1;
      for (auto s : shape) n_el *= s;
      const bool is_norm = name.find("norm") != std::string::npos;
      std::vector<double>& vals = w[name];
      const std::uint64_t begin = data.size();
      for (std::uint64_t i = 0; i < n_el; ++i) {
        float v = is_norm ? 1.0f + 0.1f * nd(rng) : 0.3f * nd(rng);
        if (tweak) v = tweak(name, i, v);
        const std::uint16_t b = f32_to_bf16(v);
        vals.push_back(static_cast<double>(bf16_to_f32(b)));
        ef_test::append_raw(data, b);
      }
      header += (header.size() > 1 ? "," : "") + std::string("\"") + name + "\":{\"dtype\":\"BF16\",\"shape\":[";
      for (std::size_t k = 0; k < shape.size(); ++k) header += (k ? "," : "") + std::to_string(shape[k]);
      header += "],\"data_offsets\":[" + std::to_string(begin) + "," + std::to_string(data.size()) + "]}";
    }
    header += "}";
    while ((8 + header.size()) % 8) header += ' ';
    const auto bytes = ef_test::make_safetensors(header, data);
    std::ofstream(path / "model.safetensors", std::ios::binary)
        .write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  }
  ~TinyDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

}  // namespace ef_test
