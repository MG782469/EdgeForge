// EdgeForge - verify-math: Stage 2 kernels vs PyTorch golden outputs.
#include "ef/verify_math.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <optional>
#include <set>
#include <vector>

#include "ef/error.h"
#include "ef/format.h"
#include "ef/golden.h"
#include "ef/kernels.h"
#include "ef/model_config.h"
#include "ef/safetensors.h"

namespace fs = std::filesystem;

namespace ef {

namespace {

struct Report {
  int pass = 0, fail = 0, skip = 0;

  void header() {
    std::printf("  %-48s %10s %10s %11s  %s\n", "check", "max|err|", "rel err", "cosine", "result");
  }
  // rel_tol: max_abs_err / max|ref|; cos_min: minimum cosine similarity.
  bool row(const std::string& name, const DiffStats& d, double rel_tol, double cos_min = 0.0) {
    const bool ok = d.non_finite == 0 && d.rel_err <= rel_tol && d.cosine >= cos_min;
    std::printf("  %-48s %10.3e %10.2e %11.9f  %s (tol %.0e)\n", name.c_str(), d.max_abs_err, d.rel_err, d.cosine,
                ok ? "PASS" : "FAIL", rel_tol);
    if (!ok && d.non_finite) std::printf("      %zu non-finite values in our output\n", d.non_finite);
    if (ok) {
      ++pass;
    } else {
      ++fail;
    }
    return ok;
  }
  void flag(const std::string& name, bool ok, const std::string& detail) {
    std::printf("  %-48s %s  %s\n", name.c_str(), detail.c_str(), ok ? "PASS" : "FAIL");
    if (ok) {
      ++pass;
    } else {
      ++fail;
    }
  }
  void skipped(const std::string& name, const std::string& why) {
    std::printf("  %-48s SKIP (%s)\n", name.c_str(), why.c_str());
    ++skip;
  }
};

// Tolerances (FP32 kernels vs FP32 PyTorch on the same inputs).
constexpr double kTolElementwise = 1e-5;  // rope, residual, swiglu, rmsnorm
constexpr double kTolReduction = 1e-4;    // GEMV / attention: different summation order than BLAS
constexpr double kCosMin = 0.99999;

struct Dims {
  std::size_t D = 0, F = 0, H = 0, KV = 0, hd = 0, kvd = 0, V = 0, L = 0;
  float eps = 1e-5f, theta = 10000.f;
};

Dims dims_from(const ModelConfig& c) {
  Dims d;
  d.D = static_cast<std::size_t>(c.hidden_size);
  d.F = static_cast<std::size_t>(c.intermediate_size);
  d.H = static_cast<std::size_t>(c.num_attention_heads);
  d.KV = static_cast<std::size_t>(c.num_key_value_heads);
  d.hd = static_cast<std::size_t>(c.head_dim);
  d.kvd = d.KV * d.hd;
  d.V = static_cast<std::size_t>(c.vocab_size);
  d.L = static_cast<std::size_t>(c.num_hidden_layers);
  d.eps = static_cast<float>(c.rms_norm_eps);
  d.theta = static_cast<float>(c.rope_theta);
  return d;
}

std::vector<float> read_all_f32(const SafetensorsFile& st, const std::string& name) {
  const TensorInfo& t = st.at(name);
  std::vector<float> v(static_cast<std::size_t>(t.numel));
  st.read_f32(t, 0, t.numel, v.data());
  return v;
}

// Applies a per-row function over S rows and compares the stacked result.
template <class Fn>
DiffStats per_row(std::size_t S, std::size_t out_dim, const GoldenTensor& ref, Fn fn) {
  std::vector<float> out(S * out_dim);
  for (std::size_t p = 0; p < S; ++p) fn(p, out.data() + p * out_dim);
  return compare(out.data(), ref.f32.data(), out.size());
}

}  // namespace

int run_verify_math(const VerifyMathOptions& opt) {
  const fs::path prompt_dir = opt.golden_dir / "prompts" / opt.prompt;
  const GoldenPrompt g = GoldenPrompt::open(prompt_dir);

  // Model dimensions: from the golden meta (always present) or the model's config.json.
  ModelConfig cfg;
  {
    const fs::path meta_path = opt.golden_dir / "meta.json";
    std::ifstream in(meta_path, std::ios::binary);
    if (!in) throw Error("cannot open '" + meta_path.string() + "'");
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const json::Value meta = json::parse(text);
    cfg = parse_model_config(meta.at("config"));
    std::printf("EdgeForge verify-math: Stage 2 kernels vs PyTorch golden (%s, reference %s, transformers %s)\n",
                opt.prompt.c_str(), meta.at("reference_dtype").as_string().c_str(),
                meta.at("versions").at("transformers").as_string().c_str());
  }
  const Dims d = dims_from(cfg);
  const std::vector<std::int32_t> ids = g.input_ids();
  const std::size_t S = ids.size();
  const bool detail = g.has("layer0/q");
  std::printf("prompt: %zu tokens | D %zu | heads %zu/%zu kv | head_dim %zu | weights: %s\n\n", S, d.D, d.H, d.KV,
              d.hd, opt.model_dir.empty() ? "not loaded (weight-free checks only)" : opt.model_dir.string().c_str());

  Report rep;
  rep.header();
  const kern::RopeTable rope(std::max<std::size_t>(S, 1), d.hd, d.theta);

  // ------------------------------------------------------------ weight-free checks
  if (detail) {
    {  // RoPE tables: golden is [S, head_dim] = cat(freqs, freqs)
      const GoldenTensor gc = g.load("rope_cos"), gs = g.load("rope_sin");
      std::vector<float> oc(S * d.hd), os(S * d.hd);
      for (std::size_t p = 0; p < S; ++p) {
        for (std::size_t i = 0; i < d.hd; ++i) {
          oc[p * d.hd + i] = rope.cos_row(p)[i % (d.hd / 2)];
          os[p * d.hd + i] = rope.sin_row(p)[i % (d.hd / 2)];
        }
      }
      DiffStats c = compare(oc.data(), gc.f32.data(), oc.size());
      const DiffStats s2 = compare(os.data(), gs.f32.data(), os.size());
      c.max_abs_err = std::max(c.max_abs_err, s2.max_abs_err);
      c.rel_err = std::max(c.rel_err, s2.rel_err);
      c.cosine = std::min(c.cosine, s2.cosine);
      rep.row("RoPE cos/sin tables", c, kTolElementwise);
    }
    const GoldenTensor gq = g.load("layer0/q"), gk = g.load("layer0/k"), gv = g.load("layer0/v");
    const GoldenTensor gqr = g.load("layer0/q_rope"), gkr = g.load("layer0/k_rope");
    // Our RoPE applied to HF's raw q/k projections.
    std::vector<float> q_r(gq.f32), k_r(gk.f32);
    for (std::size_t p = 0; p < S; ++p) {
      kern::rope_apply(q_r.data() + p * d.D, d.H, d.hd, rope.cos_row(p), rope.sin_row(p));
      kern::rope_apply(k_r.data() + p * d.kvd, d.KV, d.hd, rope.cos_row(p), rope.sin_row(p));
    }
    rep.row(strf("RoPE on q (%zu heads, half-split)", d.H), compare(q_r.data(), gqr.f32.data(), q_r.size()), kTolElementwise);
    rep.row(strf("RoPE on k (%zu KV heads)", d.KV), compare(k_r.data(), gkr.f32.data(), k_r.size()), kTolElementwise);

    {  // The real end-to-end check: HF q,k,v -> our RoPE + softmax + GQA attention -> HF context.
      const GoldenTensor ctx = g.load("layer0/attn_context");
      std::vector<float> scores(S);
      const DiffStats a = per_row(S, d.D, ctx, [&](std::size_t p, float* out) {
        kern::attention(out, q_r.data() + p * d.D, k_r.data(), gv.f32.data(), p + 1, d.H, d.KV, d.hd, scores.data());
      });
      rep.row("RoPE + causal GQA attention (softmax) vs HF", a, kTolReduction, kCosMin);
    }
    const GoldenTensor embed = g.load("embed_all"), attn_out = g.load("layer0/attn_out");
    const GoldenTensor resid = g.load("layer0/resid_after_attn");
    {
      std::vector<float> r(embed.f32);
      kern::add_inplace(r.data(), attn_out.f32.data(), r.size());
      rep.row("residual add (x + attention)", compare(r.data(), resid.f32.data(), r.size()), kTolElementwise);
    }
    const GoldenTensor gate = g.load("layer0/gate"), up = g.load("layer0/up"), act = g.load("layer0/mlp_act");
    {
      std::vector<float> o(gate.f32.size());
      kern::swiglu(o.data(), gate.f32.data(), up.f32.data(), o.size());
      rep.row("SiLU / SwiGLU: silu(gate) * up", compare(o.data(), act.f32.data(), o.size()), kTolElementwise);
    }
    {
      const GoldenTensor mlp = g.load("layer0/mlp_out"), out = g.load("layer0/out");
      std::vector<float> r(resid.f32);
      kern::add_inplace(r.data(), mlp.f32.data(), r.size());
      rep.row("residual add (x + MLP) = layer 0 output", compare(r.data(), out.f32.data(), r.size()), kTolElementwise);
    }
  } else {
    rep.skipped("layer-0 kernel checks", "prompt has no detail dumps; use --prompt p0_factual");
  }
  {
    const GoldenTensor logits = g.load("logits_last");
    const json::Value& greedy = g.manifest().at("greedy").at("ids");
    const std::size_t am = kern::argmax(logits.f32.data(), logits.f32.size());
    const auto first = greedy.size() ? greedy[0].as_uint64() : 0;
    rep.flag("argmax(logits_last) == HF greedy token", am == first,
             strf("%zu vs %llu", am, static_cast<unsigned long long>(first)));
  }

  // ------------------------------------------------------------ checks that need weights
  if (opt.model_dir.empty()) {
    rep.skipped("embedding / RMSNorm / GEMV / chained layer 0 / lm_head", "pass --model-dir to run");
  } else {
    const ModelConfig mcfg = load_model_config(opt.model_dir / "config.json");
    if (mcfg.hidden_size != cfg.hidden_size || mcfg.num_hidden_layers != cfg.num_hidden_layers ||
        mcfg.vocab_size != cfg.vocab_size) {
      throw Error("model config does not match the golden data's config");
    }
    const SafetensorsFile st = SafetensorsFile::open(opt.model_dir / "model.safetensors");
    auto W = [&](const std::string& name) { return st.bf16_data(st.at(name)); };
    const std::string L0 = "model.layers.0.";
    const std::uint16_t* emb = W("model.embed_tokens.weight");
    const std::vector<float> w_in = read_all_f32(st, L0 + "input_layernorm.weight");
    const std::vector<float> w_post = read_all_f32(st, L0 + "post_attention_layernorm.weight");
    const std::vector<float> w_final = read_all_f32(st, "model.norm.weight");
    const std::uint16_t* wq = W(L0 + "self_attn.q_proj.weight");
    const std::uint16_t* wk = W(L0 + "self_attn.k_proj.weight");
    const std::uint16_t* wv = W(L0 + "self_attn.v_proj.weight");
    const std::uint16_t* wo = W(L0 + "self_attn.o_proj.weight");
    const std::uint16_t* wg = W(L0 + "mlp.gate_proj.weight");
    const std::uint16_t* wu = W(L0 + "mlp.up_proj.weight");
    const std::uint16_t* wd = W(L0 + "mlp.down_proj.weight");
    const std::uint16_t* wlm = W("lm_head.weight");

    if (detail) {
      const GoldenTensor embed = g.load("embed_all");
      {
        const DiffStats e = per_row(S, d.D, embed, [&](std::size_t p, float* out) {
          kern::embedding_bf16(out, emb, static_cast<std::size_t>(ids[p]), d.D);
        });
        rep.row("embedding lookup (BF16 -> F32, must be exact)", e, 0.0);
      }
      const GoldenTensor in_norm = g.load("layer0/input_norm");
      rep.row("RMSNorm (input_layernorm)", per_row(S, d.D, in_norm, [&](std::size_t p, float* out) {
                kern::rmsnorm(out, embed.row(p), w_in.data(), d.D, d.eps);
              }), kTolElementwise);
      const struct {
        const char* name;
        const char* key;
        const std::uint16_t* w;
        std::size_t rows, cols;
        const char* input;
      } gemvs[] = {
          {"q_proj", "layer0/q", wq, d.D, d.D, "layer0/input_norm"},
          {"k_proj", "layer0/k", wk, d.kvd, d.D, "layer0/input_norm"},
          {"v_proj", "layer0/v", wv, d.kvd, d.D, "layer0/input_norm"},
          {"o_proj", "layer0/attn_out", wo, d.D, d.D, "layer0/attn_context"},
          {"gate_proj", "layer0/gate", wg, d.F, d.D, "layer0/post_attn_norm"},
          {"up_proj", "layer0/up", wu, d.F, d.D, "layer0/post_attn_norm"},
          {"down_proj", "layer0/mlp_out", wd, d.D, d.F, "layer0/mlp_act"},
      };
      for (const auto& gv : gemvs) {
        const GoldenTensor in = g.load(gv.input), ref = g.load(gv.key);
        rep.row(strf("GEMV %-9s [%zu x %zu]", gv.name, gv.rows, gv.cols), per_row(S, gv.rows, ref, [&](std::size_t p, float* out) {
                  kern::matvec_bf16(out, gv.w, in.row(p), gv.rows, gv.cols);
                }), kTolReduction, kCosMin);
      }
      const GoldenTensor resid = g.load("layer0/resid_after_attn"), post = g.load("layer0/post_attn_norm");
      rep.row("RMSNorm (post_attention_layernorm)", per_row(S, d.D, post, [&](std::size_t p, float* out) {
                kern::rmsnorm(out, resid.row(p), w_post.data(), d.D, d.eps);
              }), kTolElementwise);

      {  // Whole layer 0 from token ids using only our kernels (KV cache built position by position).
        const GoldenTensor ref = g.load("layer0/out");
        std::vector<float> kc(S * d.kvd), vc(S * d.kvd), out(S * d.D);
        std::vector<float> x(d.D), h(d.D), q(d.D), ctx(d.D), o(d.D), gt(d.F), u(d.F), scores(S);
        for (std::size_t p = 0; p < S; ++p) {
          kern::embedding_bf16(x.data(), emb, static_cast<std::size_t>(ids[p]), d.D);
          kern::rmsnorm(h.data(), x.data(), w_in.data(), d.D, d.eps);
          kern::matvec_bf16(q.data(), wq, h.data(), d.D, d.D);
          kern::matvec_bf16(kc.data() + p * d.kvd, wk, h.data(), d.kvd, d.D);
          kern::matvec_bf16(vc.data() + p * d.kvd, wv, h.data(), d.kvd, d.D);
          kern::rope_apply(q.data(), d.H, d.hd, rope.cos_row(p), rope.sin_row(p));
          kern::rope_apply(kc.data() + p * d.kvd, d.KV, d.hd, rope.cos_row(p), rope.sin_row(p));
          kern::attention(ctx.data(), q.data(), kc.data(), vc.data(), p + 1, d.H, d.KV, d.hd, scores.data());
          kern::matvec_bf16(o.data(), wo, ctx.data(), d.D, d.D);
          kern::add_inplace(x.data(), o.data(), d.D);
          kern::rmsnorm(h.data(), x.data(), w_post.data(), d.D, d.eps);
          kern::matvec_bf16(gt.data(), wg, h.data(), d.F, d.D);
          kern::matvec_bf16(u.data(), wu, h.data(), d.F, d.D);
          kern::swiglu(gt.data(), gt.data(), u.data(), d.F);
          kern::matvec_bf16(o.data(), wd, gt.data(), d.D, d.F);
          kern::add_inplace(x.data(), o.data(), d.D);
          std::copy(x.begin(), x.end(), out.begin() + static_cast<std::ptrdiff_t>(p * d.D));
        }
        rep.row("CHAINED layer 0 (ids -> layer output, ours only)", compare(out.data(), ref.f32.data(), out.size()),
                kTolReduction, kCosMin);
      }
      {
        const GoldenTensor hidden = g.load("hidden_all"), fnorm = g.load("final_norm_all");
        const float* last_layer = hidden.f32.data() + (d.L - 1) * S * d.D;
        std::vector<float> o(S * d.D);
        for (std::size_t p = 0; p < S; ++p) {
          kern::rmsnorm(o.data() + p * d.D, last_layer + p * d.D, w_final.data(), d.D, d.eps);
        }
        rep.row("RMSNorm (final model.norm)", compare(o.data(), fnorm.f32.data(), o.size()), kTolElementwise);
      }
    }
    {  // lm_head on the last position + top-k agreement + single-thread bandwidth.
      const GoldenTensor fn = g.load("final_norm_last"), logits = g.load("logits_last");
      std::vector<float> o(d.V);
      double best_s = 1e30;
      for (int rep_i = 0; rep_i < 3; ++rep_i) {
        const auto t0 = std::chrono::steady_clock::now();
        kern::matvec_bf16(o.data(), wlm, fn.f32.data(), d.V, d.D);
        best_s = std::min(best_s, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      }
      rep.row(strf("GEMV lm_head [%zu x %zu] -> logits", d.V, d.D), compare(o.data(), logits.f32.data(), o.size()), kTolReduction,
              kCosMin);
      auto topk = [](const std::vector<float>& v, std::size_t k) {
        std::vector<std::size_t> idx(v.size());
        for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                          [&](std::size_t a, std::size_t b) { return v[a] > v[b]; });
        return std::set<std::size_t>(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k));
      };
      const bool same_top1 = kern::argmax(o.data(), o.size()) == kern::argmax(logits.f32.data(), logits.f32.size());
      const bool same_top5 = topk(o, 5) == topk(logits.f32, 5);
      rep.flag("top-1 and top-5 tokens identical to HF", same_top1 && same_top5,
               strf("top1 %s, top5 %s", same_top1 ? "same" : "DIFF", same_top5 ? "same" : "DIFF"));
      const double bytes = static_cast<double>(d.V * d.D * 2);
      const double gbps = bytes / best_s / 1e9;
      std::printf("\n  baseline: scalar 1-thread BF16 GEMV (lm_head, %.0f MB) %.1f ms -> %.2f GB/s", bytes / 1e6,
                  best_s * 1e3, gbps);
      if (!opt.bench_json.empty()) {
        std::ifstream in(opt.bench_json, std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (!text.empty()) {
          const double peak = json::parse(text).at("peak_median_gbps").as_double();
          std::printf(" = %.0f%% of the membench baseline %.2f GB/s", 100.0 * gbps / peak, peak);
        }
      }
      std::printf("\n");
    }
  }

  std::printf("\nRESULT: %s  (%d passed, %d failed, %d skipped)\n", rep.fail ? "FAIL" : "PASS", rep.pass, rep.fail,
              rep.skip);
  return rep.fail ? 1 : 0;
}

}  // namespace ef
