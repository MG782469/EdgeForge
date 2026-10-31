// EdgeForge - Stage 3 forward-pass tests on a tiny random Llama written by the test.
// The reference recomputes the whole sequence in double precision WITHOUT a KV
// cache, so it independently checks the layer loop, RoPE positions and caching.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "ef/fp16.h"
#include "ef/kernels.h"
#include "ef/model.h"
#include "test_framework.h"
#include "test_util.h"

namespace fs = std::filesystem;
using namespace ef;

namespace {

struct Tiny {
  std::size_t D = 16, H = 4, KV = 2, hd = 4, F = 24, V = 11, L = 2;
  double eps = 1e-5, theta = 10000.0;
  bool tied = false;
};

struct TinyDir {
  fs::path path;
  std::map<std::string, std::vector<double>> w;  // exact (BF16-rounded) weights as double
  explicit TinyDir(const Tiny& t, unsigned seed = 1, bool drop_one = false) {
    static std::atomic<int> n{0};
    path = fs::temp_directory_path() / ("edgeforge_tiny_" + std::to_string(n++) + "_" +
                                        std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    fs::create_directories(path);
    std::ofstream(path / "config.json")
        << "{\"model_type\":\"llama\",\"hidden_size\":" << t.D << ",\"intermediate_size\":" << t.F
        << ",\"num_hidden_layers\":" << t.L << ",\"num_attention_heads\":" << t.H
        << ",\"num_key_value_heads\":" << t.KV << ",\"vocab_size\":" << t.V
        << ",\"rms_norm_eps\":1e-05,\"rope_theta\":10000.0,\"max_position_embeddings\":64,"
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
        const float v = is_norm ? 1.0f + 0.1f * nd(rng) : 0.3f * nd(rng);
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

// Full-sequence recompute in double, no KV cache. Returns logits[pos][V] and the
// residual stream after every layer: hidden[layer][pos][D].
struct RefOut {
  std::vector<std::vector<double>> logits;
  std::vector<std::vector<std::vector<double>>> hidden;
};

RefOut reference_forward(const Tiny& t, const std::map<std::string, std::vector<double>>& w,
                         const std::vector<std::int32_t>& ids) {
  const std::size_t S = ids.size(), kvd = t.KV * t.hd;
  using Mat = std::vector<std::vector<double>>;
  auto matvec = [](const std::vector<double>& W, const std::vector<double>& x, std::size_t rows) {
    std::vector<double> y(rows, 0.0);
    const std::size_t cols = x.size();
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t c = 0; c < cols; ++c) y[r] += W[r * cols + c] * x[c];
    return y;
  };
  auto rms = [&](const std::vector<double>& x, const std::vector<double>& g) {
    double ss = 0;
    for (double v : x) ss += v * v;
    const double inv = 1.0 / std::sqrt(ss / double(x.size()) + t.eps);
    std::vector<double> o(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) o[i] = g[i] * x[i] * inv;
    return o;
  };
  auto rope = [&](std::vector<double>& v, std::size_t heads, std::size_t pos) {
    for (std::size_t h = 0; h < heads; ++h)
      for (std::size_t i = 0; i < t.hd / 2; ++i) {
        const double ang = double(pos) * std::pow(t.theta, -2.0 * double(i) / double(t.hd));
        const double a = v[h * t.hd + i], b = v[h * t.hd + i + t.hd / 2];
        v[h * t.hd + i] = a * std::cos(ang) - b * std::sin(ang);
        v[h * t.hd + i + t.hd / 2] = b * std::cos(ang) + a * std::sin(ang);
      }
  };
  const auto& E = w.at("model.embed_tokens.weight");
  Mat X(S, std::vector<double>(t.D));
  for (std::size_t p = 0; p < S; ++p)
    for (std::size_t i = 0; i < t.D; ++i) X[p][i] = E[std::size_t(ids[p]) * t.D + i];
  RefOut out;
  for (std::size_t l = 0; l < t.L; ++l) {
    const std::string pf = "model.layers." + std::to_string(l) + ".";
    Mat Q(S), K(S), Vv(S);
    for (std::size_t p = 0; p < S; ++p) {
      const auto h = rms(X[p], w.at(pf + "input_layernorm.weight"));
      Q[p] = matvec(w.at(pf + "self_attn.q_proj.weight"), h, t.D);
      K[p] = matvec(w.at(pf + "self_attn.k_proj.weight"), h, kvd);
      Vv[p] = matvec(w.at(pf + "self_attn.v_proj.weight"), h, kvd);
      rope(Q[p], t.H, p);
      rope(K[p], t.KV, p);
    }
    for (std::size_t p = 0; p < S; ++p) {
      std::vector<double> ctx(t.D, 0.0);
      for (std::size_t hh = 0; hh < t.H; ++hh) {
        const std::size_t kv = hh / (t.H / t.KV);
        std::vector<double> s(p + 1);
        double mx = -1e300;
        for (std::size_t j = 0; j <= p; ++j) {
          double d = 0;
          for (std::size_t i = 0; i < t.hd; ++i) d += Q[p][hh * t.hd + i] * K[j][kv * t.hd + i];
          s[j] = d / std::sqrt(double(t.hd));
          mx = std::max(mx, s[j]);
        }
        double sum = 0;
        for (auto& v : s) sum += (v = std::exp(v - mx));
        for (std::size_t j = 0; j <= p; ++j)
          for (std::size_t i = 0; i < t.hd; ++i) ctx[hh * t.hd + i] += s[j] / sum * Vv[j][kv * t.hd + i];
      }
      const auto o = matvec(w.at(pf + "self_attn.o_proj.weight"), ctx, t.D);
      for (std::size_t i = 0; i < t.D; ++i) X[p][i] += o[i];
      const auto h2 = rms(X[p], w.at(pf + "post_attention_layernorm.weight"));
      auto g = matvec(w.at(pf + "mlp.gate_proj.weight"), h2, t.F);
      const auto u = matvec(w.at(pf + "mlp.up_proj.weight"), h2, t.F);
      for (std::size_t i = 0; i < t.F; ++i) g[i] = g[i] / (1.0 + std::exp(-g[i])) * u[i];
      const auto dn = matvec(w.at(pf + "mlp.down_proj.weight"), g, t.D);
      for (std::size_t i = 0; i < t.D; ++i) X[p][i] += dn[i];
    }
    out.hidden.push_back(X);
  }
  const auto& head = w.at(t.tied ? "model.embed_tokens.weight" : "lm_head.weight");
  for (std::size_t p = 0; p < S; ++p) out.logits.push_back(matvec(head, rms(X[p], w.at("model.norm.weight")), t.V));
  return out;
}

double max_rel(const float* ours, const std::vector<double>& ref) {
  double err = 0, mx = 1e-30;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    err = std::max(err, std::fabs(double(ours[i]) - ref[i]));
    mx = std::max(mx, std::fabs(ref[i]));
  }
  return err / mx;
}

const std::vector<std::int32_t> kIds = {1, 5, 9, 3, 3, 10, 0, 7, 4};

}  // namespace

TEST(model_load_validates_and_counts_bytes) {
  const Tiny t;
  const TinyDir d(t);
  const Model m = Model::load(d.path);
  CHECK_EQ(m.n_layers(), std::size_t{2});
  CHECK(m.lm_head() != m.embed_tokens());
  // per token: 2 layers x (q,k,v,o,gate,up,down + 2 norms) + final norm + lm_head, BF16
  const std::uint64_t per_layer = (16 * 16 * 2 + 8 * 16 * 2 + 24 * 16 * 3 + 16 * 2) * 2;
  CHECK_EQ(m.weight_bytes_per_token(), 2 * per_layer + 16 * 2 + 11 * 16 * 2);
  const TinyDir broken(t, 1, /*drop_one=*/true);
  CHECK_THROWS(Model::load(broken.path), "does not match config.json");
}

TEST(model_forward_matches_full_recompute_reference) {
  const Tiny t;
  const TinyDir d(t);
  const Model m = Model::load(d.path);
  const RefOut ref = reference_forward(t, d.w, kIds);
  Transformer tr(m, 16);
  ForwardTrace trace;
  trace.capture = true;
  double worst_logits = 0, worst_hidden = 0;
  for (std::size_t p = 0; p < kIds.size(); ++p) {
    const float* logits = tr.forward(kIds[p], p, true, &trace);
    worst_logits = std::max(worst_logits, max_rel(logits, ref.logits[p]));
    for (std::size_t l = 0; l < t.L; ++l)
      worst_hidden = std::max(worst_hidden, max_rel(trace.layer_out.data() + l * t.D, ref.hidden[l][p]));
    CHECK_EQ(kern::argmax(logits, t.V), static_cast<std::size_t>(std::max_element(ref.logits[p].begin(), ref.logits[p].end()) - ref.logits[p].begin()));
  }
  CHECK(worst_logits < 1e-5);
  CHECK(worst_hidden < 1e-5);
  CHECK_EQ(tr.cached(), kIds.size());
}

TEST(model_kv_cache_rewind_matches_fresh_run) {
  const Tiny t;
  const TinyDir d(t);
  const Model m = Model::load(d.path);
  Transformer a(m, 16), b(m, 16);
  for (std::size_t p = 0; p < 6; ++p) a.forward(kIds[p], p);
  const float* la = a.forward(8, 3);  // rewind: replace position 3, drop 4..5
  CHECK_EQ(a.cached(), std::size_t{4});
  std::vector<float> va(la, la + t.V);
  for (std::size_t p = 0; p < 3; ++p) b.forward(kIds[p], p);
  const float* lb = b.forward(8, 3);
  int diff = 0;
  for (std::size_t i = 0; i < t.V; ++i) diff += f32_to_bits(va[i]) != f32_to_bits(lb[i]);
  CHECK_EQ(diff, 0);  // deterministic and independent of the discarded positions
}

TEST(model_forward_rejects_bad_input) {
  const Tiny t;
  const TinyDir d(t);
  const Model m = Model::load(d.path);
  Transformer tr(m, 4);
  CHECK_THROWS(tr.forward(1, 1), "skips ahead");
  CHECK_THROWS(tr.forward(-1, 0), "outside [0, 11)");
  CHECK_THROWS(tr.forward(11, 0), "outside [0, 11)");
  for (std::size_t p = 0; p < 4; ++p) tr.forward(1, p, false);
  CHECK_THROWS(tr.forward(1, 4), "KV-cache capacity");
  CHECK(tr.forward(1, 3, false) == nullptr);
  CHECK_THROWS(Transformer(m, 65), "max_position_embeddings");
  CHECK_THROWS(Transformer(m, 0), "must be > 0");
}

TEST(model_generate_greedy) {
  const Tiny t;
  const TinyDir d(t);
  const Model m = Model::load(d.path);
  Transformer tr(m, 32);
  const std::vector<std::int32_t> prompt(kIds.begin(), kIds.begin() + 5);
  const GreedyResult g = generate_greedy(tr, prompt, 6, -1, 3);
  CHECK_EQ(g.tokens.size(), std::size_t{6});
  CHECK_EQ(g.step_logits.size(), std::size_t{3});
  CHECK_EQ(g.decode_forwards, std::size_t{5});
  // Same tokens via the reference: greedy over full recomputation.
  std::vector<std::int32_t> seq = prompt;
  for (std::size_t s = 0; s < 6; ++s) {
    const auto ref = reference_forward(t, d.w, seq);
    const auto& last = ref.logits.back();
    const auto tok = static_cast<std::int32_t>(std::max_element(last.begin(), last.end()) - last.begin());
    CHECK_EQ(g.tokens[s], tok);
    seq.push_back(tok);
  }
  const GreedyResult stop = generate_greedy(tr, prompt, 6, g.tokens[0], 0);
  CHECK_EQ(stop.tokens.size(), std::size_t{1});  // EOS ends generation
  CHECK_THROWS(generate_greedy(tr, std::vector<std::int32_t>(30, 1), 3, -1), "exceeds the KV-cache capacity");
  CHECK_THROWS(generate_greedy(tr, {}, 3, -1), "empty prompt");
}

TEST(model_tied_embeddings) {
  Tiny t;
  t.tied = true;
  const TinyDir d(t, 7);
  const Model m = Model::load(d.path);
  CHECK(m.lm_head() == m.embed_tokens());
  const RefOut ref = reference_forward(t, d.w, kIds);
  Transformer tr(m, 16);
  const float* logits = nullptr;
  for (std::size_t p = 0; p < kIds.size(); ++p) logits = tr.forward(kIds[p], p);
  CHECK(max_rel(logits, ref.logits.back()) < 1e-5);
}
