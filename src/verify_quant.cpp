// EdgeForge - Stage 5: verify-quant and quant-analyze.
#include "ef/verify_quant.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <thread>
#include <vector>

#include "ef/error.h"
#include "ef/format.h"
#include "ef/fp16.h"
#include "ef/golden.h"
#include "ef/kernels.h"
#include "ef/quantizer.h"

namespace fs = std::filesystem;

namespace ef {

namespace {

constexpr double kTolQ4 = 0.12;   // relative RMSE bound per Q4_0 tensor (typical ~0.08)
constexpr double kTolQ8 = 0.01;   // typical ~0.004
constexpr double kCosLayer = 0.99;  // GEMV outputs with dequantized weights vs exact BF16 weights
// Re-derived activations vs the golden layer outputs: same tolerance as verify-forward.
constexpr double kTolHidden = 1e-3, kCosHidden = 0.9999;

struct Checker {
  VerifyQuantReport* r;
  void report(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (ok) {
      ++r->passed;
    } else {
      ++r->failed;
      r->failed_checks.push_back(what);
    }
  }
};

template <class Fn>
void parallel_for(std::uint64_t n, unsigned threads, Fn fn) {
  const unsigned T = std::max(1u, std::min<unsigned>(threads, static_cast<unsigned>(std::max<std::uint64_t>(n, 1))));
  if (T == 1) {
    fn(std::uint64_t{0}, n);
    return;
  }
  std::vector<std::thread> pool;
  const std::uint64_t per = (n + T - 1) / T;
  for (unsigned k = 0; k < T; ++k) {
    const std::uint64_t b = std::min<std::uint64_t>(n, k * per), e = std::min<std::uint64_t>(n, b + per);
    if (b < e) pool.emplace_back([&fn, b, e] { fn(b, e); });
  }
  for (auto& t : pool) t.join();
}

std::vector<float> read_matrix(const SafetensorsFile& st, const std::string& name) {
  const TensorInfo& t = st.at(name);
  std::vector<float> v(static_cast<std::size_t>(t.numel));
  st.read_f32(t, 0, t.numel, v.data());
  return v;
}

std::vector<float> dequant_matrix(const QuantFile& q, const QTensor& t) {
  std::vector<float> v(static_cast<std::size_t>(t.rows() * t.cols()));
  for (std::uint64_t r = 0; r < t.rows(); ++r) q.dequantize_row(t, r, v.data() + r * t.cols());
  return v;
}

std::set<std::size_t> topk(const float* v, std::size_t n, std::size_t k) {
  std::vector<std::size_t> idx(n);
  for (std::size_t i = 0; i < n; ++i) idx[i] = i;
  std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                    [&](std::size_t a, std::size_t b) { return v[a] > v[b]; });
  return {idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k)};
}

// GEMV of every golden input row with W (row-major [rows][cols]) vs a golden output.
DiffStats gemv_vs_golden(const std::vector<float>& W, std::size_t rows, std::size_t cols, const GoldenTensor& in,
                         const GoldenTensor& ref) {
  const std::size_t S = static_cast<std::size_t>(in.shape.at(0));
  std::vector<float> out(S * rows);
  for (std::size_t p = 0; p < S; ++p) kern::matvec_f32(out.data() + p * rows, W.data(), in.row(p), rows, cols);
  return compare(out.data(), ref.f32.data(), out.size());
}

struct LayerGemv {
  const char* proj;
  const char* in;
  const char* out;
};
constexpr LayerGemv kLayer0[] = {
    {"self_attn.q_proj", "layer0/input_norm", "layer0/q"},       {"self_attn.k_proj", "layer0/input_norm", "layer0/k"},
    {"self_attn.v_proj", "layer0/input_norm", "layer0/v"},       {"self_attn.o_proj", "layer0/attn_context", "layer0/attn_out"},
    {"mlp.gate_proj", "layer0/post_attn_norm", "layer0/gate"},   {"mlp.up_proj", "layer0/post_attn_norm", "layer0/up"},
    {"mlp.down_proj", "layer0/mlp_act", "layer0/mlp_out"},
};

// ---------------------------------------------------------------- activations for every layer
//
// The golden data holds the residual stream entering every layer (embeddings + the output of
// each decoder layer), for all positions of the detailed prompt and for the last position of
// the others. From it we re-derive, with the exact BF16 weights and the verified Stage 2
// kernels, the input of every projection of every layer:
//   kind 0  attn_in = RMSNorm(x)                 -> q_proj, k_proj, v_proj   (all prompts)
//   kind 1  ctx     = attention(RoPE(q), RoPE(k), v)  -> o_proj               (full prompts)
//   kind 2  mlp_in  = RMSNorm(x + o)             -> gate_proj, up_proj        (full prompts)
//   kind 3  act     = SiLU(gate) * up            -> down_proj                 (full prompts)
// Each layer starts from the golden input, so nothing accumulates, and recomputing the layer
// output must reproduce the golden one (checked: proves the activations belong to this model).

struct ProjSpec {
  const char* name;
  int kind;
};
constexpr ProjSpec kProjs[] = {{"self_attn.q_proj", 0}, {"self_attn.k_proj", 0}, {"self_attn.v_proj", 0},
                               {"self_attn.o_proj", 1}, {"mlp.gate_proj", 2},    {"mlp.up_proj", 2},
                               {"mlp.down_proj", 3}};

struct PromptActs {
  std::string name;
  std::size_t S = 0;   // positions
  bool full = false;   // all positions (all four kinds) vs last position (kind 0 only)
  std::vector<std::array<std::vector<float>, 4>> in;  // [layer][kind] -> [S][dim]
};

struct ActBank {
  std::vector<PromptActs> prompts;
  std::vector<std::string> lines;  // one per golden prompt, in directory order (used or skipped + why)
  std::size_t full_prompts = 0, compared_positions = 0;
  double layer_rel_err = 0.0, layer_cos = 1.0;  // re-derived layer outputs vs golden (full prompts)
};

std::string layer_name(std::size_t l, const char* proj) {
  return "model.layers." + std::to_string(l) + "." + proj + ".weight";
}

// Y[s][r] = W[r] . X[s] for every position s; rows are split across threads and each weight row
// is reused for all positions while it is in cache. Per dot product exactly kern::matvec_f32.
void matmul_positions(float* Y, const std::vector<float>& W, std::size_t rows, std::size_t cols, const float* X,
                      std::size_t S, unsigned threads) {
  parallel_for(rows, threads, [&](std::uint64_t b, std::uint64_t e) {
    for (std::size_t r = static_cast<std::size_t>(b); r < e; ++r) {
      for (std::size_t s = 0; s < S; ++s) Y[s * rows + r] = kern::dot_f32(W.data() + r * cols, X + s * cols, cols);
    }
  });
}

ActBank build_activations(const fs::path& golden_dir, const SafetensorsFile& st, const ModelConfig& c,
                          unsigned threads) {
  ActBank b;
  const auto D = static_cast<std::size_t>(c.hidden_size), F = static_cast<std::size_t>(c.intermediate_size);
  const auto H = static_cast<std::size_t>(c.num_attention_heads), KV = static_cast<std::size_t>(c.num_key_value_heads);
  const auto hd = static_cast<std::size_t>(c.head_dim), kvd = KV * hd;
  const auto L = static_cast<std::size_t>(c.num_hidden_layers);
  const auto eps = static_cast<float>(c.rms_norm_eps);
  std::vector<std::vector<float>> attn_norm(L), ffn_norm(L);
  for (std::size_t l = 0; l < L; ++l) {
    attn_norm[l] = read_matrix(st, "model.layers." + std::to_string(l) + ".input_layernorm.weight");
    ffn_norm[l] = read_matrix(st, "model.layers." + std::to_string(l) + ".post_attention_layernorm.weight");
  }
  std::vector<fs::path> dirs;
  if (fs::is_directory(golden_dir / "prompts")) {
    for (const auto& e : fs::directory_iterator(golden_dir / "prompts")) {
      if (fs::exists(e.path() / "manifest.json")) dirs.push_back(e.path());
    }
  }
  std::sort(dirs.begin(), dirs.end());
  auto mismatch = [](const std::string& p) {
    return Error("golden prompt '" + p + "': activation shapes do not match this model (different model?)");
  };

  std::vector<PromptActs> full_list;
  for (const auto& dir : dirs) {
    const GoldenPrompt g = GoldenPrompt::open(dir);
    PromptActs pa;
    pa.name = dir.filename().string();
    GoldenTensor E, Hs;
    const bool has_full = g.has("embed_all") && g.has("hidden_all");
    const bool has_last = g.has("embed_last") && g.has("hidden_last");
    if (!has_full && !has_last) {
      b.lines.push_back(strf("%-14s skipped: no residual-stream data", pa.name.c_str()));
      continue;
    }
    try {
      E = g.load(has_full ? "embed_all" : "embed_last");
      Hs = g.load(has_full ? "hidden_all" : "hidden_last");
    } catch (const Error& err) {
      b.lines.push_back(strf("%-14s skipped: %s", pa.name.c_str(), err.what()));
      continue;
    }
    pa.full = has_full;
    pa.S = has_full ? static_cast<std::size_t>(E.shape.at(0)) : 1;
    pa.in.resize(L);
    if (has_full) {
      if (E.shape != std::vector<std::uint64_t>{pa.S, D} || Hs.shape != std::vector<std::uint64_t>{L, pa.S, D}) {
        throw mismatch(pa.name);
      }
    } else if (E.numel() != D || Hs.shape != std::vector<std::uint64_t>{L, D}) {
      throw mismatch(pa.name);
    }
    const std::size_t S = pa.S;
    std::vector<float> h(S * D);
    if (!has_full) {
      for (std::size_t l = 0; l < L; ++l) {
        const float* x = l == 0 ? E.f32.data() : Hs.f32.data() + (l - 1) * D;
        kern::rmsnorm(h.data(), x, attn_norm[l].data(), D, eps);
        pa.in[l][0] = h;
      }
      b.lines.push_back(strf("%-14s last position -> inputs of q/k/v", pa.name.c_str()));
      b.prompts.push_back(std::move(pa));
      continue;
    }
    // Full prompt: every position, every projection input, layer by layer from the golden input.
    const kern::RopeTable rope(S, hd, static_cast<float>(c.rope_theta));
    std::vector<float> q(S * D), k(S * kvd), v(S * kvd), ctx(S * D), o(S * D), r(S * D), h2(S * D);
    std::vector<float> gate(S * F), up(S * F), act(S * F), down(S * D), out(S * D);
    for (std::size_t l = 0; l < L; ++l) {
      const float* X = l == 0 ? E.f32.data() : Hs.f32.data() + (l - 1) * S * D;
      auto mm = [&](float* Y, const char* proj, std::size_t rows, std::size_t cols, const float* in) {
        matmul_positions(Y, read_matrix(st, layer_name(l, proj)), rows, cols, in, S, threads);
      };
      for (std::size_t s = 0; s < S; ++s) kern::rmsnorm(h.data() + s * D, X + s * D, attn_norm[l].data(), D, eps);
      mm(q.data(), "self_attn.q_proj", D, D, h.data());
      mm(k.data(), "self_attn.k_proj", kvd, D, h.data());
      mm(v.data(), "self_attn.v_proj", kvd, D, h.data());
      for (std::size_t s = 0; s < S; ++s) {
        kern::rope_apply(q.data() + s * D, H, hd, rope.cos_row(s), rope.sin_row(s));
        kern::rope_apply(k.data() + s * kvd, KV, hd, rope.cos_row(s), rope.sin_row(s));
      }
      parallel_for(S, threads, [&](std::uint64_t s0, std::uint64_t s1) {
        std::vector<float> scores(S);
        for (std::size_t s = static_cast<std::size_t>(s0); s < s1; ++s) {
          kern::attention(ctx.data() + s * D, q.data() + s * D, k.data(), v.data(), s + 1, H, KV, hd, scores.data());
        }
      });
      mm(o.data(), "self_attn.o_proj", D, D, ctx.data());
      for (std::size_t i = 0; i < S * D; ++i) r[i] = X[i] + o[i];
      for (std::size_t s = 0; s < S; ++s) kern::rmsnorm(h2.data() + s * D, r.data() + s * D, ffn_norm[l].data(), D, eps);
      mm(gate.data(), "mlp.gate_proj", F, D, h2.data());
      mm(up.data(), "mlp.up_proj", F, D, h2.data());
      for (std::size_t s = 0; s < S; ++s) kern::swiglu(act.data() + s * F, gate.data() + s * F, up.data() + s * F, F);
      mm(down.data(), "mlp.down_proj", D, F, act.data());
      for (std::size_t i = 0; i < S * D; ++i) out[i] = r[i] + down[i];
      const DiffStats d = compare(out.data(), Hs.f32.data() + l * S * D, S * D);
      b.layer_rel_err = std::max(b.layer_rel_err, d.non_finite ? 1e30 : d.rel_err);
      b.layer_cos = std::min(b.layer_cos, d.cosine);
      pa.in[l] = {h, ctx, h2, act};
    }
    ++b.full_prompts;
    b.compared_positions += S;
    b.lines.push_back(strf("%-14s %zu positions -> inputs of all 7 projections", pa.name.c_str(), S));
    full_list.push_back(std::move(pa));
  }
  // Full prompts first (they feed every projection), then the last-position ones.
  b.prompts.insert(b.prompts.begin(), std::make_move_iterator(full_list.begin()),
                   std::make_move_iterator(full_list.end()));
  return b;
}

struct InputSet {
  const std::string* prompt;
  const float* X;  // [S][cols]
  std::size_t S;
};

std::vector<InputSet> inputs_for(const ActBank& b, std::size_t layer, int kind) {
  std::vector<InputSet> v;
  for (const auto& p : b.prompts) {
    const auto& x = p.in.at(layer)[static_cast<std::size_t>(kind)];
    if (!x.empty()) v.push_back({&p.name, x.data(), p.S});
  }
  return v;
}

// y = W x for every input position with the exact source weights and with candidate weights
// (cand(row, source_row, out) writes the candidate row), compared per prompt.
template <class Cand>
GemvCheck measure_gemv(const SafetensorsFile& st, const TensorInfo& src, const std::vector<InputSet>& ins,
                       unsigned threads, Cand cand) {
  GemvCheck g;
  g.tensor = src.name;
  const std::uint64_t rows = src.shape.at(0), cols = src.shape.at(1);
  std::vector<const float*> xs;
  for (const auto& in : ins) {
    for (std::size_t s = 0; s < in.S; ++s) xs.push_back(in.X + s * cols);
  }
  const std::size_t T = xs.size();
  std::vector<float> yr(T * rows), yq(T * rows);
  parallel_for(rows, threads, [&](std::uint64_t b, std::uint64_t e) {
    std::vector<float> a(static_cast<std::size_t>(cols)), w(static_cast<std::size_t>(cols));
    for (std::uint64_t r = b; r < e; ++r) {
      st.read_f32(src, r * cols, cols, a.data());
      cand(r, a.data(), w.data());
      for (std::size_t t = 0; t < T; ++t) {
        yr[t * rows + r] = kern::dot_f32(a.data(), xs[t], static_cast<std::size_t>(cols));
        yq[t * rows + r] = kern::dot_f32(w.data(), xs[t], static_cast<std::size_t>(cols));
      }
    }
  });
  std::size_t t0 = 0;
  for (const auto& in : ins) {
    double dot = 0, na = 0, nb = 0, se = 0;
    for (std::size_t s = 0; s < in.S; ++s, ++t0) {
      double td = 0, ta = 0, tb = 0;
      for (std::uint64_t r = 0; r < rows; ++r) {
        const double a = static_cast<double>(yq[t0 * rows + r]), bb = static_cast<double>(yr[t0 * rows + r]);
        td += a * bb;
        ta += a * a;
        tb += bb * bb;
        se += (a - bb) * (a - bb);
      }
      dot += td;
      na += ta;
      nb += tb;
      const double tc = (ta > 0 && tb > 0) ? td / std::sqrt(ta * tb) : (ta == tb ? 1.0 : 0.0);
      g.token_cosine = std::min(g.token_cosine, tc);
    }
    const double cs = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : (na == nb ? 1.0 : 0.0);
    if (g.prompts == 0 || cs < g.cosine) {
      g.cosine = cs;
      g.rel_l2 = nb > 0 ? std::sqrt(se / nb) : 0.0;
      g.prompt = *in.prompt;
    }
    ++g.prompts;
    g.positions += in.S;
  }
  return g;
}

// Candidate rows: re-quantize the source row to `type` (exactly what the quantizer would store).
auto requant(QType type, std::uint64_t cols, bool search) {
  return [type, cols, search](std::uint64_t, const float* src, float* out) {
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(qtype_row_bytes(type, cols)));
    quantize_row(type, src, static_cast<std::size_t>(cols), buf.data(), search);
    dequantize_row(type, buf.data(), static_cast<std::size_t>(cols), out);
  };
}

double weight_rel_rmse(const SafetensorsFile& st, const TensorInfo& s, QType type, bool search, unsigned threads) {
  const std::uint64_t rows = s.shape.size() == 2 ? s.shape[0] : 1, cols = s.shape.back();
  std::vector<double> sse(rows), ref(rows);
  parallel_for(rows, threads, [&](std::uint64_t b, std::uint64_t e) {
    std::vector<float> a(static_cast<std::size_t>(cols)), d(static_cast<std::size_t>(cols));
    const auto cand = requant(type, cols, search);
    for (std::uint64_t r = b; r < e; ++r) {
      st.read_f32(s, r * cols, cols, a.data());
      cand(r, a.data(), d.data());
      double se = 0, rs = 0;
      for (std::uint64_t k = 0; k < cols; ++k) {
        const double err = static_cast<double>(a[k]) - static_cast<double>(d[k]);
        se += err * err;
        rs += static_cast<double>(a[k]) * static_cast<double>(a[k]);
      }
      sse[r] = se;
      ref[r] = rs;
    }
  });
  double a = 0, b = 0;
  for (std::uint64_t r = 0; r < rows; ++r) {
    a += sse[r];
    b += ref[r];
  }
  return b > 0 ? std::sqrt(a / b) : 0.0;
}

double rmse_bound(QType t) { return t == QType::Q4_0 ? kTolQ4 : t == QType::Q8_0 ? kTolQ8 : 0.0; }

// Next precision step up (Q4_0 -> Q8_0 -> BF16).
QType step_up(QType t) { return t == QType::Q4_0 ? QType::Q8_0 : QType::BF16; }

std::string signed_bytes(std::int64_t v) {
  return (v < 0 ? "-" : "+") + human_bytes(static_cast<std::uint64_t>(v < 0 ? -v : v));
}

}  // namespace

// ============================================================================ verify-quant

int run_verify_quant(const VerifyQuantOptions& o) {
  const unsigned threads = o.threads ? o.threads : std::max(1u, std::thread::hardware_concurrency());
  VerifyQuantReport local;
  VerifyQuantReport& R = o.report != nullptr ? *o.report : local;
  R = VerifyQuantReport{};
  std::printf("EdgeForge verify-quant: %s vs %s\n", o.qmodel.string().c_str(), o.model_dir.string().c_str());
  Checker chk{&R};
  QuantFile q;
  try {
    q = QuantFile::open(o.qmodel);
    chk.report(true, strf("file structure valid: %zu tensors, 64-byte aligned, no overlap, sizes match types, "
                          "every type explained by the policy",
                          q.tensors().size()));
  } catch (const Error& e) {
    chk.report(false, e.what());
    std::printf("RESULT: FAIL\n");
    return 1;
  }
  const ModelConfig src_cfg = load_model_config(o.model_dir / "config.json");
  const SafetensorsFile st = SafetensorsFile::open(o.model_dir / "model.safetensors");
  const ModelConfig& c = q.config();
  chk.report(c.hidden_size == src_cfg.hidden_size && c.intermediate_size == src_cfg.intermediate_size &&
                 c.num_hidden_layers == src_cfg.num_hidden_layers && c.num_attention_heads == src_cfg.num_attention_heads &&
                 c.num_key_value_heads == src_cfg.num_key_value_heads && c.vocab_size == src_cfg.vocab_size &&
                 c.rms_norm_eps == src_cfg.rms_norm_eps && c.rope_theta == src_cfg.rope_theta &&
                 c.tie_word_embeddings == src_cfg.tie_word_embeddings,
             "embedded config.json matches the source checkpoint");
  const json::Value& qs = q.header().at("quant");
  const QuantPolicy& policy = q.policy();
  std::printf("  settings: layers %s, lm_head %s, embed %s, norms %s, block %lld, scale %s\n",
              qtype_name(policy.layers), qtype_name(policy.lm_head), qtype_name(policy.embed),
              qs.at("norms").as_string().c_str(), static_cast<long long>(qs.at("block").as_int64()),
              qs.at("scale").as_string().c_str());
  std::printf("  policy   : %s\n", policy.name.empty() ? "(unnamed)" : policy.name.c_str());
  if (policy.overrides.empty()) {
    std::printf("  overrides: none\n");
  } else {
    for (const auto& ov : policy.overrides) {
      std::printf("  override : %s -> %s\n", ov.pattern.c_str(), qtype_name(ov.type));
    }
  }
  const bool search = qs.at("scale").as_string() == "mse-search";

  // Per-tensor: recompute error, exactness, determinism on sampled rows.
  struct TypeAgg {
    std::size_t count = 0;
    std::uint64_t params = 0, bytes = 0;
    double sse = 0, ref = 0, worst = 0;
    std::string worst_name;
  };
  std::map<std::string, TypeAgg> agg;
  std::map<std::string, double> measured_rel;  // tensor -> recomputed rel RMSE
  std::size_t exact_ok = 0, exact_total = 0, stats_ok = 0, nonfinite = 0, det_rows = 0, det_ok = 0;
  std::vector<std::string> problems;
  const auto t0 = std::chrono::steady_clock::now();
  for (const QTensor& t : q.tensors()) {
    const TensorInfo& s = st.at(t.name);
    const std::uint64_t rows = t.rows(), cols = t.cols();
    std::vector<double> sse(rows), ref(rows);
    std::vector<std::uint64_t> bad(rows, 0), nf(rows, 0);
    parallel_for(rows, threads, [&](std::uint64_t b, std::uint64_t e) {
      std::vector<float> a(static_cast<std::size_t>(cols)), d(static_cast<std::size_t>(cols));
      for (std::uint64_t r = b; r < e; ++r) {
        st.read_f32(s, r * cols, cols, a.data());
        q.dequantize_row(t, r, d.data());
        double se = 0, rs = 0;
        for (std::uint64_t k = 0; k < cols; ++k) {
          if (!std::isfinite(d[k])) ++nf[r];
          if (f32_to_bits(a[k]) != f32_to_bits(d[k])) ++bad[r];
          const double err = static_cast<double>(a[k]) - static_cast<double>(d[k]);
          se += err * err;
          rs += static_cast<double>(a[k]) * static_cast<double>(a[k]);
        }
        sse[r] = se;
        ref[r] = rs;
      }
    });
    double tsse = 0, tref = 0;
    std::uint64_t tbad = 0;
    for (std::uint64_t r = 0; r < rows; ++r) {
      tsse += sse[r];
      tref += ref[r];
      tbad += bad[r];
      nonfinite += nf[r];
    }
    const double rel = tref > 0 ? std::sqrt(tsse / tref) : 0.0;
    measured_rel[t.name] = rel;
    if (!qtype_is_block(t.type)) {  // stored losslessly
      ++exact_total;
      if (tbad == 0) {
        ++exact_ok;
      } else {
        problems.push_back(t.name + ": " + std::to_string(tbad) + " values differ from the source");
      }
    }
    if (std::fabs(rel - t.rel_rmse) <= 1e-5 * std::max(rel, 1e-12) + 1e-12) {
      ++stats_ok;
    } else if (problems.size() < 10) {
      problems.push_back(strf("%s: stored rel_rmse %.6e, recomputed %.6e", t.name.c_str(), t.rel_rmse, rel));
    }
    // determinism: re-quantize a few rows from the source and compare bytes
    const std::uint64_t row_bytes = qtype_row_bytes(t.type, cols);
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(row_bytes));
    const std::uint64_t n_samples = std::min<std::uint64_t>(rows, 16);
    for (std::uint64_t k = 0; k < n_samples; ++k) {
      const std::uint64_t r = n_samples == 1 ? 0 : k * (rows - 1) / (n_samples - 1);
      quantize_tensor_rows(st, s, t.type, search, r, r + 1, buf.data());
      ++det_rows;
      if (std::memcmp(buf.data(), q.row_data(t, r), static_cast<std::size_t>(row_bytes)) == 0) ++det_ok;
    }
    TypeAgg& a = agg[qtype_name(t.type)];
    ++a.count;
    a.params += rows * cols;
    a.bytes += t.bytes;
    a.sse += tsse;
    a.ref += tref;
    if (rel >= a.worst) {
      a.worst = rel;
      a.worst_name = t.name;
    }
  }
  const double scan_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  chk.report(exact_ok == exact_total, strf("lossless tensors (embed/norms) bit-identical to the source: %zu/%zu",
                                           exact_ok, exact_total));
  chk.report(nonfinite == 0, "no NaN/Inf after dequantization");
  chk.report(stats_ok == q.tensors().size(),
             strf("per-tensor error stored in the header matches recomputation: %zu/%zu", stats_ok, q.tensors().size()));
  chk.report(det_ok == det_rows, strf("deterministic: %zu/%zu re-quantized sample rows byte-identical to the file",
                                      det_ok, det_rows));
  std::printf("\n  %-6s %7s %14s %12s %16s %12s  worst tensor\n", "type", "tensors", "params", "size", "rel RMSE (all)",
              "worst");
  for (const auto& [name, a] : agg) {
    std::printf("  %-6s %7zu %14s %12s %16.4e %12.4e  %s\n", name.c_str(), a.count, with_commas(a.params).c_str(),
                human_bytes(a.bytes).c_str(), a.ref > 0 ? std::sqrt(a.sse / a.ref) : 0.0, a.worst, a.worst_name.c_str());
  }
  {  // the five largest Q4_0 errors: shows how close the bound is, not just the maximum
    std::vector<std::pair<double, std::string>> q4;
    for (const auto& t : q.tensors()) {
      if (t.type == QType::Q4_0) q4.emplace_back(measured_rel[t.name], t.name);
    }
    std::sort(q4.rbegin(), q4.rend());
    if (!q4.empty()) std::printf("\n  largest Q4_0 weight errors (bound %.2f):\n", kTolQ4);
    for (std::size_t i = 0; i < std::min<std::size_t>(5, q4.size()); ++i) {
      std::printf("    %.4f  %s\n", q4[i].first, q4[i].second.c_str());
    }
  }
  std::printf("\n");
  if (agg.count("Q4_0")) {
    chk.report(agg["Q4_0"].worst <= kTolQ4, strf("Q4_0 reconstruction: worst tensor rel RMSE %.4f <= %.2f (%s)",
                                                 agg["Q4_0"].worst, kTolQ4, agg["Q4_0"].worst_name.c_str()));
  }
  if (agg.count("Q8_0")) {
    chk.report(agg["Q8_0"].worst <= kTolQ8, strf("Q8_0 reconstruction: worst tensor rel RMSE %.4f <= %.2f (%s)",
                                                 agg["Q8_0"].worst, kTolQ8, agg["Q8_0"].worst_name.c_str()));
  }
  for (const auto& p : problems) std::printf("         %s\n", p.c_str());

  std::uint64_t src_bytes = 0, src_per_tok = 0;
  for (const auto& t : q.tensors()) {
    const auto n = st.at(t.name).nbytes();
    src_bytes += n;
    if (t.name != "model.embed_tokens.weight" || c.tie_word_embeddings) src_per_tok += n;
  }
  std::printf("\n  size on disk : %s -> %s  (%.2fx smaller)\n", human_bytes(src_bytes).c_str(),
              human_bytes(q.file_size()).c_str(), static_cast<double>(src_bytes) / static_cast<double>(q.file_size()));
  std::printf("  read / token : %s -> %s  (%.2fx less weight traffic per decoded token; decode speed follows\n"
              "                 it when decoding is limited by memory traffic - measured in Stages 6/7)\n",
              human_bytes(src_per_tok).c_str(), human_bytes(q.bytes_per_token()).c_str(),
              static_cast<double>(src_per_tok) / static_cast<double>(q.bytes_per_token()));

  // Overrides versus the uniform policy: weight error and cost (activation error added below).
  std::int64_t override_extra = 0;
  for (const auto& t : q.tensors()) {
    if (t.rule != "override") continue;
    OverrideCheck oc;
    oc.tensor = t.name;
    oc.stored = t.type;
    oc.baseline = policy_choice(policy, t.name, t.cols(), false).type;
    oc.rmse_stored = measured_rel[t.name];
    oc.rmse_baseline = weight_rel_rmse(st, st.at(t.name), oc.baseline, search, threads);
    oc.extra_bytes = static_cast<std::int64_t>(t.bytes) -
                     static_cast<std::int64_t>(qtype_row_bytes(oc.baseline, t.cols()) * t.rows());
    override_extra += oc.extra_bytes;
    R.overrides.push_back(oc);
  }
  if (!R.overrides.empty()) {
    std::printf("  overrides    : %s of tensor data vs the uniform policy (%+.3f%% file, %+.3f%% read/token)\n",
                signed_bytes(override_extra).c_str(),
                100.0 * static_cast<double>(override_extra) / static_cast<double>(q.file_size() - override_extra),
                100.0 * static_cast<double>(override_extra) / static_cast<double>(q.bytes_per_token() - override_extra));
  }
  std::printf("  (full reconstruction scan: %.1f s, %u threads)\n", scan_s, threads);

  // ---------------- activation-level checks on real PyTorch activations
  std::map<std::string, const GemvCheck*> gemv_by_name;
  ActBank bank;
  if (!o.golden_dir.empty()) {
    std::printf("\n  output error with quantized weights on real activations (golden):\n");
    const fs::path p0 = o.golden_dir / "prompts" / "p0_factual";
    bool layer0_done = false;
    if (fs::exists(p0 / "manifest.json")) {
      const GoldenPrompt g = GoldenPrompt::open(p0);
      if (g.has("layer0/input_norm")) {
        double worst_cos = 1.0;
        std::string worst;
        try {
          std::printf("    layer 0 vs the PyTorch layer-0 outputs (p0_factual, all positions):\n");
          for (const auto& L : kLayer0) {
            const QTensor& t = q.at(layer_name(0, L.proj));
            const GoldenTensor in = g.load(L.in), ref = g.load(L.out);
            if (in.row_size() != t.cols() || ref.row_size() != t.rows()) {
              throw Error("golden layer-0 data does not match this model's shapes (different model?)");
            }
            const DiffStats d = gemv_vs_golden(dequant_matrix(q, t), static_cast<std::size_t>(t.rows()),
                                               static_cast<std::size_t>(t.cols()), in, ref);
            if (d.cosine < worst_cos) {
              worst_cos = d.cosine;
              worst = L.proj;
            }
            std::printf("      %-18s %-5s rel err %.4f  cosine %.6f\n", L.proj, qtype_name(t.type), d.rel_err, d.cosine);
          }
          layer0_done = true;
        } catch (const Error& err) {
          if (std::string(err.what()).find("different model") != std::string::npos) throw;
          std::printf("      skipped: %s\n", err.what());
        }
        if (layer0_done) {
          chk.report(worst_cos >= kCosLayer, strf("layer-0 GEMV outputs keep cosine >= %.2f vs PyTorch (worst %.6f, %s)",
                                                  kCosLayer, worst_cos, worst.c_str()));
        }
      }
    }

    // Every projection of every layer, on activations re-derived from the golden residual stream.
    const auto tb = std::chrono::steady_clock::now();
    std::printf("\n    activations for every layer, re-derived from the golden residual stream (exact BF16 weights):\n");
    bank = build_activations(o.golden_dir, st, c, threads);
    for (const auto& line : bank.lines) std::printf("      %s\n", line.c_str());
    for (const auto& p : bank.prompts) R.prompts_used.push_back(p.name);
    bool acts_ok = true;
    if (bank.full_prompts > 0) {
      R.activation_rel_err = bank.layer_rel_err;
      acts_ok = bank.layer_rel_err <= kTolHidden && bank.layer_cos >= kCosHidden;
      chk.report(acts_ok, strf("re-derived activations reproduce the golden layer outputs (%lld layers x %zu positions: "
                               "max rel err %.2e <= %.0e, cosine %.6f)",
                               static_cast<long long>(c.num_hidden_layers), bank.compared_positions,
                               bank.layer_rel_err, kTolHidden, bank.layer_cos));
    }
    if (!acts_ok) {
      // Output errors measured on activations of another model would be meaningless.
      chk.report(false, "layer matrices on real activations: not measured, the golden activations do not belong "
                        "to this model");
    } else if (!bank.prompts.empty()) {
      const auto L = static_cast<std::size_t>(c.num_hidden_layers);
      for (std::size_t l = 0; l < L; ++l) {
        for (const auto& P : kProjs) {
          const auto ins = inputs_for(bank, l, P.kind);
          if (ins.empty()) continue;
          const QTensor& t = q.at(layer_name(l, P.name));
          GemvCheck g = measure_gemv(st, st.at(t.name), ins, threads,
                                     [&](std::uint64_t r, const float*, float* out) { q.dequantize_row(t, r, out); });
          g.type = t.type;
          R.gemv.push_back(std::move(g));
        }
      }
      for (const auto& g : R.gemv) gemv_by_name[g.tensor] = &g;
      std::vector<const GemvCheck*> order;
      for (const auto& g : R.gemv) order.push_back(&g);
      std::stable_sort(order.begin(), order.end(), [](const GemvCheck* a, const GemvCheck* b) { return a->cosine < b->cosine; });
      std::size_t below = 0;
      for (const auto* g : order) below += g->cosine < kCosLayer;
      const std::size_t show = std::min(order.size(), std::max<std::size_t>(10, std::min<std::size_t>(below, 30)));
      std::printf("\n    %zu layer matrices, worst %zu by GEMV cosine (min over prompts; 'worst pos' = single position):\n",
                  order.size(), show);
      std::printf("      %-42s %-5s %10s %9s %10s  %s\n", "tensor", "type", "cosine", "rel L2", "worst pos", "prompt");
      for (std::size_t i = 0; i < show; ++i) {
        const GemvCheck& g = *order[i];
        std::printf("      %-42s %-5s %10.6f %9.4f %10.6f  %s%s\n", g.tensor.c_str(), qtype_name(g.type), g.cosine,
                    g.rel_l2, g.token_cosine, g.prompt.c_str(), g.cosine < kCosLayer ? "  <-- below bound" : "");
      }
      if (below > show) std::printf("      ... %zu more below the bound\n", below - show);
      std::printf("      worst per projection over all layers:");
      for (const auto& P : kProjs) {
        double w = 1.0;
        for (const auto& g : R.gemv) {
          if (g.tensor.find(std::string(".") + P.name + ".") != std::string::npos) w = std::min(w, g.cosine);
        }
        std::printf(" %s %.4f", std::strchr(P.name, '.') + 1, w);
      }
      std::printf("\n");
      const double worst = order.empty() ? 1.0 : order.front()->cosine;
      const std::size_t total = L * std::size(kProjs);
      if (order.size() < total) {
        std::printf("      note: only %zu of %zu layer matrices have activations (o/gate/up/down need a prompt with all "
                    "positions)\n",
                    order.size(), total);
      }
      chk.report(below == 0,
                 strf("layer matrices keep GEMV cosine >= %.2f on every prompt: %zu/%zu measured pass (worst %.6f, %s)",
                      kCosLayer, order.size() - below, order.size(), worst,
                      order.empty() ? "-" : order.front()->tensor.c_str()));
      std::printf("      note: the bound is per matrix and per prompt, over all of that prompt's positions together.\n"
                  "            It is not a per-position guarantee ('worst pos' can be lower) and not an end-to-end\n"
                  "            one: errors of 154 matrices compound through 22 layers (measured in Stage 6).\n");
      std::printf("      (activation checks: %.1f s)\n",
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - tb).count());
    }
  }

  // Overrides: what each exception buys (weight error, output error) and what it costs.
  if (!R.overrides.empty()) {
    std::printf("\n  overrides vs the uniform policy (same tensor, both precisions measured):\n");
    std::printf("    %-42s %-14s %-19s %-23s %s\n", "tensor", "type", "weight rel RMSE", "GEMV cosine (min)", "extra");
    for (auto& oc : R.overrides) {
      const auto it = gemv_by_name.find(oc.tensor);
      if (it != gemv_by_name.end()) {
        oc.cos_stored = it->second->cosine;
        const TensorInfo& s = st.at(oc.tensor);
        const int kind = [&] {
          for (const auto& P : kProjs) {
            if (oc.tensor.find(std::string(".") + P.name + ".") != std::string::npos) return P.kind;
          }
          return 0;
        }();
        const std::size_t layer = static_cast<std::size_t>(std::stoul(oc.tensor.substr(13)));
        oc.cos_baseline = measure_gemv(st, s, inputs_for(bank, layer, kind), threads,
                                       requant(oc.baseline, s.shape.at(1), search)).cosine;
      }
      const std::string cosines =
          oc.cos_stored >= 0 ? strf("%.6f -> %.6f", oc.cos_baseline, oc.cos_stored) : std::string("(no activations)");
      std::printf("    %-42s %-14s %-19s %-23s %s\n", oc.tensor.c_str(),
                  strf("%s -> %s", qtype_name(oc.baseline), qtype_name(oc.stored)).c_str(),
                  strf("%.4f -> %.4f", oc.rmse_baseline, oc.rmse_stored).c_str(), cosines.c_str(),
                  signed_bytes(oc.extra_bytes).c_str());
    }
  }

  // Anything that fails a bound: measure the next precision up and print a ready --override.
  {
    std::map<std::string, std::string> reasons;
    for (const auto& t : q.tensors()) {
      if (qtype_is_block(t.type) && measured_rel[t.name] > rmse_bound(t.type)) {
        reasons[t.name] = strf("weight rel RMSE %.4f > %.2f", measured_rel[t.name], rmse_bound(t.type));
      }
    }
    for (const auto& g : R.gemv) {
      if (g.cosine < kCosLayer) {
        auto& r = reasons[g.tensor];
        r += (r.empty() ? "" : ", ") + strf("GEMV cosine %.4f < %.2f", g.cosine, kCosLayer);
      }
    }
    for (const auto& [name, why] : reasons) {
      const QTensor& t = q.at(name);
      const TensorInfo& s = st.at(name);
      Promotion p;
      p.tensor = name;
      p.reason = why;
      p.from = t.type;
      p.rmse_from = measured_rel[name];
      const auto it = gemv_by_name.find(name);
      p.cos_from = it != gemv_by_name.end() ? it->second->cosine : -1.0;
      p.to = t.type;
      while (p.to != QType::BF16) {
        p.to = step_up(p.to);
        p.rmse_to = p.to == QType::BF16 ? 0.0 : weight_rel_rmse(st, s, p.to, search, threads);
        p.cos_to = -1.0;
        if (it != gemv_by_name.end()) {
          const std::size_t layer = static_cast<std::size_t>(std::stoul(name.substr(13)));
          int kind = 0;
          for (const auto& P : kProjs) {
            if (name.find(std::string(".") + P.name + ".") != std::string::npos) kind = P.kind;
          }
          p.cos_to = measure_gemv(st, s, inputs_for(bank, layer, kind), threads, requant(p.to, t.cols(), search)).cosine;
        }
        if (p.rmse_to <= rmse_bound(p.to) && (p.cos_to < 0 || p.cos_to >= kCosLayer)) break;
      }
      p.extra_bytes = static_cast<std::int64_t>(qtype_row_bytes(p.to, t.cols()) * t.rows()) -
                      static_cast<std::int64_t>(t.bytes);
      R.promotions.push_back(p);
    }
    if (!R.promotions.empty()) {
      std::printf("\n  failing tensors, re-measured one precision step up (nothing is changed automatically):\n");
      std::int64_t extra = 0;
      std::vector<QuantOverride> sugg = policy.overrides;
      for (const auto& p : R.promotions) {
        const std::string cos = p.cos_from >= 0 ? strf("  cosine %.6f -> %.6f", p.cos_from, p.cos_to) : "";
        std::printf("    %-42s %s -> %s  rel RMSE %.4f -> %.4f%s  %s   [%s]\n", p.tensor.c_str(), qtype_name(p.from),
                    qtype_name(p.to), p.rmse_from, p.rmse_to, cos.c_str(), signed_bytes(p.extra_bytes).c_str(),
                    p.reason.c_str());
        extra += p.extra_bytes;
        sugg.push_back({p.tensor, p.to});
      }
      R.suggested_override = overrides_to_arg(sugg);
      std::printf("  to quantize with these promotions (%s on top of this file; the list below is complete):\n",
                  signed_bytes(extra).c_str());
      std::printf("    edgeforge quantize --model-dir DIR --out FILE.efq --policy uniform --override %s\n",
                  R.suggested_override.c_str());
    }
  }

  // lm_head on every prompt's final hidden state
  if (!o.golden_dir.empty()) {
    std::size_t top1 = 0, prompts = 0, top5_overlap = 0;
    const QTensor& head = q.at(c.tie_word_embeddings ? "model.embed_tokens.weight" : "lm_head.weight");
    const std::vector<float> Wh = dequant_matrix(q, head);
    std::vector<fs::path> dirs;
    if (fs::is_directory(o.golden_dir / "prompts")) {
      for (const auto& e : fs::directory_iterator(o.golden_dir / "prompts")) {
        if (fs::exists(e.path() / "manifest.json")) dirs.push_back(e.path());
      }
    }
    std::sort(dirs.begin(), dirs.end());
    std::printf("\n");
    for (const auto& dir : dirs) {
      const GoldenPrompt g = GoldenPrompt::open(dir);
      if (!g.has("final_norm_last")) continue;
      GoldenTensor fn, lg;
      try {
        fn = g.load("final_norm_last");
        lg = g.load("logits_last");
      } catch (const Error& err) {
        std::printf("    lm_head %-12s skipped: %s\n", dir.filename().string().c_str(), err.what());
        continue;
      }
      if (fn.f32.size() != static_cast<std::size_t>(head.cols()) || lg.f32.size() != static_cast<std::size_t>(head.rows())) {
        std::printf("    lm_head %-12s skipped: golden data is for a different model\n", dir.filename().string().c_str());
        continue;
      }
      std::vector<float> out(static_cast<std::size_t>(head.rows()));
      kern::matvec_f32(out.data(), Wh.data(), fn.f32.data(), out.size(), static_cast<std::size_t>(head.cols()));
      ++prompts;
      top1 += kern::argmax(out.data(), out.size()) == kern::argmax(lg.f32.data(), lg.f32.size());
      const std::size_t k5 = std::min<std::size_t>(5, out.size());
      const auto a = topk(out.data(), out.size(), k5), b = topk(lg.f32.data(), lg.f32.size(), k5);
      for (auto x : a) top5_overlap += b.count(x);
      const DiffStats d = compare(out.data(), lg.f32.data(), out.size());
      std::printf("    lm_head %-12s %s  logits rel err %.4f  cosine %.6f\n", dir.filename().string().c_str(),
                  qtype_name(head.type), d.rel_err, d.cosine);
    }
    if (prompts > 0) {
      chk.report(top1 == prompts, strf("lm_head (%s): next-token top-1 identical on %zu/%zu prompts, top-5 overlap %zu/%zu",
                                       qtype_name(head.type), top1, prompts, top5_overlap,
                                       prompts * std::min<std::size_t>(5, static_cast<std::size_t>(head.rows()))));
    }
  }
  std::printf("\nRESULT: %s  (%d passed, %d failed)\n", R.failed ? "FAIL" : "PASS", R.passed, R.failed);
  return R.failed ? 1 : 0;
}

// ============================================================================ quant-analyze

int run_quant_analyze(const QuantAnalyzeOptions& o) {
  const ModelConfig cfg = load_model_config(o.model_dir / "config.json");
  const SafetensorsFile st = SafetensorsFile::open(o.model_dir / "model.safetensors");
  struct Method {
    const char* name;
    double bits;
    std::size_t group;
    bool search;
    bool q8;
  };
  const Method methods[] = {
      {"Q4 g32  absmax (llama.cpp Q4_0)", 4.5, 32, false, false},
      {"Q4 g32  MSE scale search (ours)", 4.5, 32, true, false},
      {"Q4 g64  MSE scale search", 4.25, 64, true, false},
      {"Q4 g128 MSE scale search", 4.125, 128, true, false},
      {"Q8 g32  (Q8_0)", 8.5, 32, false, true},
  };
  auto fake = [](const Method& m, const float* x, std::size_t n, float* out) {
    for (std::size_t b = 0; b < n; b += m.group) {
      const std::size_t len = std::min(m.group, n - b);
      if (m.q8) {
        BlockQ8_0 blk;
        quantize_block_q8_0(x + b, blk);
        dequantize_block_q8_0(blk, out + b);
      } else {
        fake_quant_q4_group(x + b, len, out + b, m.search);
      }
    }
  };
  std::vector<std::string> sample;
  const auto L = cfg.num_hidden_layers;
  for (std::int64_t l : {std::int64_t{0}, L / 2, L - 1}) {
    for (const char* p : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"}) {
      sample.push_back("model.layers." + std::to_string(l) + "." + p + ".weight");
    }
  }
  std::uint64_t layer_params = 0;
  for (const auto& e : expected_tensors(cfg)) {
    if (e.name.rfind("model.layers.", 0) == 0 && e.name.find("norm") == std::string::npos) {
      layer_params += e.shape[0] * e.shape[1];
    }
  }
  std::printf("EdgeForge quant-analyze: quantization error by method (%zu sampled matrices, <=256 rows each)\n\n",
              sample.size());
  std::printf("  %-34s %7s %14s %16s\n", "method", "bits/w", "layer weights", "weight rel RMSE");
  for (const Method& m : methods) {
    double sse = 0, ref = 0;
    for (const auto& name : sample) {
      const TensorInfo& t = st.at(name);
      const std::uint64_t rows = t.shape[0], cols = t.shape[1];
      const std::uint64_t step = std::max<std::uint64_t>(1, rows / 256);
      std::vector<float> a(static_cast<std::size_t>(cols)), d(static_cast<std::size_t>(cols));
      for (std::uint64_t r = 0; r < rows; r += step) {
        st.read_f32(t, r * cols, cols, a.data());
        fake(m, a.data(), a.size(), d.data());
        for (std::size_t k = 0; k < a.size(); ++k) {
          const double e = static_cast<double>(a[k]) - static_cast<double>(d[k]);
          sse += e * e;
          ref += static_cast<double>(a[k]) * static_cast<double>(a[k]);
        }
      }
    }
    std::printf("  %-34s %7.3f %14s %16.4f\n", m.name, m.bits,
                human_bytes(static_cast<std::uint64_t>(static_cast<double>(layer_params) * m.bits / 8.0)).c_str(),
                std::sqrt(sse / ref));
  }
  if (!o.golden_dir.empty() && fs::exists(o.golden_dir / "prompts" / "p0_factual" / "manifest.json")) {
    const GoldenPrompt g = GoldenPrompt::open(o.golden_dir / "prompts" / "p0_factual");
    std::printf("\n  layer-0 GEMV output error on real activations (all 7 matrices, worst cosine / mean rel err):\n");
    for (const Method& m : methods) {
      double worst_cos = 1.0, rel_sum = 0.0;
      for (const auto& Lg : kLayer0) {
        const std::string name = std::string("model.layers.0.") + Lg.proj + ".weight";
        const TensorInfo& t = st.at(name);
        std::vector<float> W = read_matrix(st, name), Wq(W.size());
        const std::size_t rows = static_cast<std::size_t>(t.shape[0]), cols = static_cast<std::size_t>(t.shape[1]);
        for (std::size_t r = 0; r < rows; ++r) fake(m, W.data() + r * cols, cols, Wq.data() + r * cols);
        const DiffStats d = gemv_vs_golden(Wq, rows, cols, g.load(Lg.in), g.load(Lg.out));
        worst_cos = std::min(worst_cos, d.cosine);
        rel_sum += d.rel_err;
      }
      std::printf("  %-34s cosine %.6f   rel err %.4f\n", m.name, worst_cos, rel_sum / 7.0);
    }
  }
  return 0;
}

}  // namespace ef
