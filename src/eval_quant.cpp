// EdgeForge - Stage 6: eval-quant.
#include "ef/eval_quant.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <random>

#include "ef/error.h"
#include "ef/format.h"
#include "ef/golden.h"
#include "ef/kernels.h"
#include "ef/model.h"
#include "ef/qkernels.h"
#include "ef/sysinfo.h"
#include "ef/threadpool.h"
#include "ef/tokenizer.h"

namespace fs = std::filesystem;

namespace ef {

namespace {

// ---- targets, fixed before the first real-model run (see README) --------------------------
constexpr double kKernelTol = 1e-5;          // packed vs dequantized dot, relative to sum |w*x|
constexpr double kTargetMeanKL = 0.10;       // mean KL(reference || quantized), nats per position
constexpr double kConfidentP = 0.5;          // the reference puts >= 50% on its top-1 token ...
constexpr double kTargetConfidentTop1 = 0.95;  // ... and we agree with it there at least 95% of the time
constexpr double kTargetPplRatio = 1.15;     // perplexity(quantized) / perplexity(BF16) on the text

struct Checker {
  EvalQuantReport* r;
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

// Next-token distribution comparison of reference logits and ours, in double precision.
struct DistDiff {
  double kl = 0.0;  // KL(P_ref || P_ours), nats
  bool top1 = false, confident = false, finite = true;
};

double log_sum_exp(const float* x, std::size_t n) {
  double m = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < n; ++i) m = std::max(m, static_cast<double>(x[i]));
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) s += std::exp(static_cast<double>(x[i]) - m);
  return m + std::log(s);
}

DistDiff compare_dist(const float* ref, const float* ours, std::size_t V) {
  DistDiff d;
  for (std::size_t i = 0; i < V; ++i) d.finite = d.finite && std::isfinite(ours[i]);
  if (!d.finite) return d;
  const double lr = log_sum_exp(ref, V), lq = log_sum_exp(ours, V);
  for (std::size_t i = 0; i < V; ++i) {
    const double a = static_cast<double>(ref[i]) - lr, b = static_cast<double>(ours[i]) - lq;
    d.kl += std::exp(a) * (a - b);
  }
  d.kl = std::max(d.kl, 0.0);
  const std::size_t ar = kern::argmax(ref, V);
  d.top1 = ar == kern::argmax(ours, V);
  d.confident = std::exp(static_cast<double>(ref[ar]) - lr) >= kConfidentP;
  return d;
}

struct DistAgg {
  std::size_t n = 0, top1 = 0, confident = 0, confident_top1 = 0, nonfinite = 0;
  double kl_sum = 0.0, kl_max = 0.0;
  void add(const DistDiff& d) {
    ++n;
    if (!d.finite) {
      ++nonfinite;
      return;
    }
    top1 += d.top1;
    confident += d.confident;
    confident_top1 += d.confident && d.top1;
    kl_sum += d.kl;
    kl_max = std::max(kl_max, d.kl);
  }
  void merge(const DistAgg& o) {
    n += o.n;
    top1 += o.top1;
    confident += o.confident;
    confident_top1 += o.confident_top1;
    nonfinite += o.nonfinite;
    kl_sum += o.kl_sum;
    kl_max = std::max(kl_max, o.kl_max);
  }
  double kl_mean() const { return n > nonfinite ? kl_sum / static_cast<double>(n - nonfinite) : 0.0; }
  std::string summary() const {
    return strf("top-1 %zu/%zu, confident top-1 %zu/%zu, KL mean %.4f max %.4f", top1, n, confident_top1, confident,
                kl_mean(), kl_max);
  }
};

double neg_log_prob(const float* logits, std::size_t V, std::int32_t target) {
  return log_sum_exp(logits, V) - static_cast<double>(logits[static_cast<std::size_t>(target)]);
}

std::string quoted(const Tokenizer* tok, const std::vector<std::int32_t>& ids) {
  if (tok == nullptr) {
    std::string s;
    for (auto id : ids) s += (s.empty() ? "" : " ") + std::to_string(id);
    return "[" + s + "]";
  }
  std::string t = tok->decode(ids, true);
  for (char& c : t) {
    if (c == '\n') c = ' ';
  }
  return "\"" + t + "\"";
}

struct KernelCheck {
  std::size_t rows = 0;
  double worst = 0.0;
  std::string worst_name;
};

// Packed dot products of sampled rows of every quantized matrix vs the exact dequantized dot.
KernelCheck check_kernels(const Model& m) {
  KernelCheck kc;
  std::mt19937 rng(6);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto one = [&](const std::string& name, const WeightView& w) {
    if (w.type != QType::Q4_0 && w.type != QType::Q8_0) return;
    std::vector<float> x(w.cols), deq(w.cols), y(w.rows);
    for (auto& v : x) v = nd(rng);
    const std::size_t samples = std::min<std::size_t>(w.rows, 16);
    for (std::size_t k = 0; k < samples; ++k) {
      const std::size_t r = samples == 1 ? 0 : k * (w.rows - 1) / (samples - 1);
      const std::uint8_t* row = w.data + r * w.row_bytes();
      kern::matvec_rows(y.data(), w, x.data(), r, r + 1);  // the kernels of the active backend
      const float got = y[r];
      dequantize_row(w.type, row, w.cols, deq.data());
      double exact = 0.0, mag = 0.0;
      for (std::size_t c = 0; c < w.cols; ++c) {
        exact += static_cast<double>(deq[c]) * static_cast<double>(x[c]);
        mag += std::fabs(static_cast<double>(deq[c]) * static_cast<double>(x[c]));
      }
      const double rel = std::isfinite(got) ? std::fabs(static_cast<double>(got) - exact) / std::max(mag, 1e-30) : 1e30;
      if (rel > kc.worst) {
        kc.worst = rel;
        kc.worst_name = name;
      }
      ++kc.rows;
    }
  };
  for (std::size_t l = 0; l < m.n_layers(); ++l) {
    const LayerWeights& w = m.layer(l);
    const std::string p = "model.layers." + std::to_string(l) + ".";
    one(p + "self_attn.q_proj", w.wq);
    one(p + "self_attn.k_proj", w.wk);
    one(p + "self_attn.v_proj", w.wv);
    one(p + "self_attn.o_proj", w.wo);
    one(p + "mlp.gate_proj", w.w_gate);
    one(p + "mlp.up_proj", w.w_up);
    one(p + "mlp.down_proj", w.w_down);
  }
  one("lm_head", m.lm_head());
  return kc;
}

}  // namespace

int run_eval_quant(const EvalQuantOptions& o) {
  EvalQuantReport local;
  EvalQuantReport& R = o.report != nullptr ? *o.report : local;
  R = EvalQuantReport{};
  Checker chk{&R};
  const Model qm = Model::load_efq(o.qmodel);
  const ModelConfig& c = qm.config();
  const auto V = static_cast<std::size_t>(c.vocab_size);
  const auto eos = static_cast<std::int32_t>(c.eos_token_id);
  std::printf("EdgeForge eval-quant: the engine on packed weights (%s kernels, %u thread%s)\n",
              kern::backend_name(kern::backend()), o.pool != nullptr ? o.pool->size() : 1u,
              o.pool != nullptr && o.pool->size() > 1 ? "s" : "");
  std::printf("  model : %s\n  file  : %s (%s), %s of weights read per token\n", qm.description().c_str(),
              o.qmodel.string().c_str(), human_bytes(fs::file_size(o.qmodel)).c_str(),
              human_bytes(qm.weight_bytes_per_token()).c_str());

  std::unique_ptr<Tokenizer> tok;
  const fs::path tok_dir = !o.tokenizer_dir.empty() ? o.tokenizer_dir : o.model_dir;
  if (!tok_dir.empty() && fs::exists(tok_dir / "tokenizer.json")) {
    tok = std::make_unique<Tokenizer>(Tokenizer::load(tok_dir));
    if (tok->vocab_size() != V) throw Error("tokenizer vocab does not match the model");
  }

  // ---- 1. kernels on the real weights
  const KernelCheck kc = check_kernels(qm);
  R.kernel_rows = kc.rows;
  R.kernel_max_rel = kc.worst;
  chk.report(kc.worst <= kKernelTol,
             strf("packed Q4_0/Q8_0 dot (%s) = dequantize-then-dot on the real weights: %zu sampled rows, max rel "
                  "err %.1e <= %.0e%s",
                  kern::backend_name(kern::backend()), kc.rows, kc.worst, kKernelTol, kc.worst_name.empty() ? "" : (" (" + kc.worst_name + ")").c_str()));

  DistAgg golden, text;
  // ---- 2. golden prompts, teacher-forced, vs PyTorch FP32
  if (!o.golden_dir.empty()) {
    std::vector<fs::path> dirs;
    if (fs::is_directory(o.golden_dir / "prompts")) {
      for (const auto& e : fs::directory_iterator(o.golden_dir / "prompts")) {
        if (fs::exists(e.path() / "manifest.json")) dirs.push_back(e.path());
      }
    }
    std::sort(dirs.begin(), dirs.end());
    std::printf("\n  next-token distributions vs the PyTorch FP32 reference (teacher-forced), greedy continuation:\n");
    for (const auto& dir : dirs) {
      const GoldenPrompt g = GoldenPrompt::open(dir);
      const std::string name = dir.filename().string();
      std::vector<std::int32_t> ids, gold;
      GoldenTensor all, last, dec;
      try {
        ids = g.input_ids();
        if (g.has("logits_all")) all = g.load("logits_all");
        last = g.load("logits_last");
        if (g.has("decode_logits")) dec = g.load("decode_logits");
      } catch (const Error& e) {
        std::printf("    %-13s skipped: %s\n", name.c_str(), e.what());
        continue;
      }
      if (last.f32.size() != V) throw Error("golden prompt '" + name + "' is for a different vocabulary");
      if (const auto* gr = g.manifest().find("greedy")) {
        for (const auto& v : gr->at("ids").items()) gold.push_back(static_cast<std::int32_t>(v.as_int64()));
      }
      const std::size_t S = ids.size();
      const std::size_t n_dec = dec.f32.empty() ? 0 : static_cast<std::size_t>(dec.shape.at(0));
      DistAgg pa;
      Transformer tr(qm, S + gold.size() + 1, o.pool);
      std::vector<float> cur(V);
      for (std::size_t p = 0; p < S; ++p) {
        const bool want = !all.f32.empty() || p + 1 == S;
        const float* lg = tr.forward(ids[p], p, want);
        if (!want) continue;
        pa.add(compare_dist(!all.f32.empty() ? all.row(p) : last.f32.data(), lg, V));
        if (p + 1 == S) std::copy_n(lg, V, cur.data());
      }
      // Teacher-forced through the golden greedy tokens; remember where our argmax first differs.
      std::vector<std::int32_t> ours;
      std::size_t div = gold.size();
      std::vector<float> div_logits;
      for (std::size_t k = 0; k < gold.size(); ++k) {
        const auto a = static_cast<std::int32_t>(kern::argmax(cur.data(), V));
        if (a != gold[k] && div == gold.size()) {
          div = k;
          div_logits = cur;
        }
        if (gold[k] == eos || k + 1 == gold.size()) break;
        const float* lg = tr.forward(gold[k], S + k, true);
        std::copy_n(lg, V, cur.data());
        if (k < n_dec) pa.add(compare_dist(dec.row(k), lg, V));
      }
      // Free-running greedy: identical up to `div`, then continue from our own token (rewinds the cache).
      ours.assign(gold.begin(), gold.begin() + static_cast<std::ptrdiff_t>(div));
      if (div < gold.size()) {
        auto t = static_cast<std::int32_t>(kern::argmax(div_logits.data(), V));
        ours.push_back(t);
        std::size_t pos = S + div;
        while (ours.size() < gold.size() && t != eos) {
          const float* lg = tr.forward(t, pos++, true);
          t = static_cast<std::int32_t>(kern::argmax(lg, V));
          ours.push_back(t);
        }
      }
      ++R.greedy_prompts;
      R.greedy_identical += ours == gold;
      std::printf("    %-13s %2zu positions: %s\n", name.c_str(), pa.n, pa.summary().c_str());
      if (!gold.empty()) {
        if (ours == gold) {
          std::printf("    %-13s greedy %zu tokens identical: %s\n", "", gold.size(), quoted(tok.get(), ours).c_str());
        } else {
          std::printf("    %-13s greedy differs from token %zu: ours %s\n", "", div, quoted(tok.get(), ours).c_str());
          std::printf("    %-13s %*s reference %s\n", "", static_cast<int>(std::to_string(div).size()) + 20, "",
                      quoted(tok.get(), gold).c_str());
        }
      }
      golden.merge(pa);
    }
  }

  // ---- 3. text perplexity: BF16 engine vs quantized engine
  std::vector<std::int32_t> text_ids = o.text_ids;
  if (text_ids.empty() && !o.text_file.empty()) {
    if (!tok) {
      std::printf("\n  text perplexity skipped: no tokenizer (--tokenizer-dir or --model-dir with tokenizer.json)\n");
    } else {
      std::ifstream in(o.text_file, std::ios::binary);
      if (!in) throw Error("cannot open text file '" + o.text_file.string() + "'");
      const std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      text_ids = tok->encode(body, true, false);  // plain text: "<s>" etc. are not special (as llama-perplexity)
    }
  }
  // Chunks to evaluate: ids, and the first position whose next-token prediction is scored.
  struct Chunk {
    std::vector<std::int32_t> ids;
    std::size_t first = 0;  // predictions at positions [first, ids.size() - 1) are scored
  };
  std::vector<Chunk> chunks;
  if (o.window == 0) {  // one window from the start, every prediction scored
    if (text_ids.size() > o.max_text_tokens) text_ids.resize(o.max_text_tokens);
    if (text_ids.size() >= 2) chunks.push_back({text_ids, 0});
  } else {  // llama-perplexity scheme: chunks of W tokens, first token -> BOS, second half scored
    const std::size_t W = o.window;
    if (W < 4) throw Error("--window must be at least 4 tokens");
    std::size_t n_chunk = text_ids.size() / W;
    if (o.max_chunks > 0) n_chunk = std::min(n_chunk, o.max_chunks);
    if (n_chunk == 0 && !text_ids.empty()) {
      std::printf("\n  text has %zu tokens: too short for --window %zu, skipped\n", text_ids.size(), W);
    }
    for (std::size_t ci = 0; ci < n_chunk; ++ci) {
      Chunk ch;
      ch.ids.assign(text_ids.begin() + static_cast<std::ptrdiff_t>(ci * W),
                    text_ids.begin() + static_cast<std::ptrdiff_t>((ci + 1) * W));
      if (c.bos_token_id >= 0) ch.ids[0] = static_cast<std::int32_t>(c.bos_token_id);
      ch.first = W / 2;
      chunks.push_back(std::move(ch));
    }
  }
  if (!chunks.empty()) {
    std::size_t max_len = 0, n_scored = 0;
    for (const auto& ch : chunks) {
      max_len = std::max(max_len, ch.ids.size());
      n_scored += ch.ids.size() - 1 - ch.first;
    }
    const std::string src = o.text_file.empty() ? std::string("(token ids)") : o.text_file.filename().string();
    if (o.window == 0) {
      std::printf("\n  text: %s, %zu tokens, teacher-forced, every prediction scored\n", src.c_str(), max_len);
    } else {
      std::printf("\n  text: %s, %zu tokens -> %zu chunks x %zu (llama-perplexity scheme: BOS first, second half "
                  "scored), %zu predictions\n",
                  src.c_str(), text_ids.size(), chunks.size(), o.window, n_scored);
    }
    std::unique_ptr<Model> ref;
    std::unique_ptr<Transformer> tb;
    Transformer tq(qm, max_len, o.pool);
    double nll_q = 0.0, nll_r = 0.0, fwd_q = 0.0, fwd_r = 0.0, worst_ratio = 0.0;
    std::size_t n_fwd_q = 0, n_fwd_r = 0;
    auto timed = [](Transformer& t, std::int32_t token, std::size_t pos, bool want, double& acc) {
      const auto f0 = std::chrono::steady_clock::now();
      const float* lg = t.forward(token, pos, want);
      acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - f0).count();
      return lg;
    };
    for (std::size_t ci = 0; ci < chunks.size(); ++ci) {
      const Chunk& ch = chunks[ci];
      const std::size_t n = ch.ids.size(), S = n - 1 - ch.first;
      std::vector<float> qlog(S * V);
      double cq = 0.0, cr = 0.0;
      tq.reset();
      for (std::size_t p = 0; p + 1 < n; ++p) {
        const bool want = p >= ch.first;
        const float* lg = timed(tq, ch.ids[p], p, want, fwd_q);
        ++n_fwd_q;
        if (!want) continue;
        std::copy_n(lg, V, qlog.data() + (p - ch.first) * V);
        cq += neg_log_prob(lg, V, ch.ids[p + 1]);
      }
      nll_q += cq;
      if (ci == 0) {
        R.q_peak_rss = peak_rss_bytes();  // before the BF16 reference is loaded
        if (!o.model_dir.empty()) {
          ref = std::make_unique<Model>(Model::load(o.model_dir));
          const ModelConfig& rc = ref->config();
          if (rc.hidden_size != c.hidden_size || rc.num_hidden_layers != c.num_hidden_layers ||
              rc.vocab_size != c.vocab_size) {
            throw Error("--model-dir is a different model than the .efq file");
          }
          tb = std::make_unique<Transformer>(*ref, max_len, o.pool);
        }
      }
      if (tb) {
        tb->reset();
        for (std::size_t p = 0; p + 1 < n; ++p) {
          const bool want = p >= ch.first;
          const float* lg = timed(*tb, ch.ids[p], p, want, fwd_r);
          ++n_fwd_r;
          if (!want) continue;
          cr += neg_log_prob(lg, V, ch.ids[p + 1]);
          text.add(compare_dist(lg, qlog.data() + (p - ch.first) * V, V));
        }
        nll_r += cr;
        worst_ratio = std::max(worst_ratio, std::exp((cq - cr) / static_cast<double>(S)));
      }
      if (chunks.size() > 1) {
        std::printf("    chunk %3zu/%zu: PPL %s%.3f%s\n", ci + 1, chunks.size(),
                    tb ? strf("%.3f -> ", std::exp(cr / static_cast<double>(S))).c_str() : "",
                    std::exp(cq / static_cast<double>(S)), tb ? " (BF16 -> quantized)" : " (quantized)");
        std::fflush(stdout);
      }
    }
    R.tps_q = static_cast<double>(n_fwd_q) / fwd_q;
    R.ppl_q = std::exp(nll_q / static_cast<double>(n_scored));
    std::printf("    quantized engine: PPL %.3f, %.2f forward passes/s\n", R.ppl_q, R.tps_q);
    if (tb) {
      R.tps_ref = static_cast<double>(n_fwd_r) / fwd_r;
      R.ppl_ref = std::exp(nll_r / static_cast<double>(n_scored));
      std::printf("    BF16 engine     : PPL %.3f, %.2f forward passes/s\n", R.ppl_ref, R.tps_ref);
      std::printf("    per position vs BF16: %s\n", text.summary().c_str());
      std::printf("    perplexity: %.3f -> %.3f (%+.2f%%)%s\n", R.ppl_ref, R.ppl_q, 100.0 * (R.ppl_q / R.ppl_ref - 1.0),
                  chunks.size() > 1 ? strf("; worst chunk ratio %.4f", worst_ratio).c_str() : "");
      const double gb_q = R.tps_q * static_cast<double>(qm.weight_bytes_per_token()) / 1e9;
      const double gb_r = R.tps_ref * static_cast<double>(ref->weight_bytes_per_token()) / 1e9;
      std::printf("\n  speed (%s kernels, %u thread%s; the full ladder: edgeforge bench-decode):\n",
                  kern::backend_name(kern::backend()), o.pool != nullptr ? o.pool->size() : 1u,
                  o.pool != nullptr && o.pool->size() > 1 ? "s" : "");
      std::printf("    BF16 engine     : %.2f tok/s, %s/token -> %.2f GB/s of weights\n", R.tps_ref,
                  human_bytes(ref->weight_bytes_per_token()).c_str(), gb_r);
      std::printf("    quantized engine: %.2f tok/s, %s/token -> %.2f GB/s of weights (%.2fx the BF16 tok/s)\n",
                  R.tps_q, human_bytes(qm.weight_bytes_per_token()).c_str(), gb_q, R.tps_q / R.tps_ref);
    } else {
      std::printf("    (no --model-dir: BF16 reference, KL and perplexity ratio not measured)\n");
    }
    std::printf("  memory: quantized engine peak RSS %s (measured before the BF16 reference is loaded)\n",
                human_bytes(R.q_peak_rss).c_str());
  }

  // ---- 4. verdict
  DistAgg allp = golden;
  allp.merge(text);
  R.golden_positions = golden.n;
  R.text_positions = text.n;
  R.positions = allp.n;
  R.top1 = allp.top1;
  R.confident = allp.confident;
  R.confident_top1 = allp.confident_top1;
  R.nonfinite = allp.nonfinite;
  R.kl_mean = allp.kl_mean();
  R.kl_max = allp.kl_max;
  std::printf("\n  all compared positions: %zu (%zu golden vs PyTorch, %zu text vs BF16 engine)\n", allp.n, golden.n,
              text.n);
  if (allp.n > 0) {
    std::printf("    top-1 agreement %zu/%zu = %.1f%% (reported, not a target: near-ties flip under any noise)\n",
                allp.top1, allp.n, 100.0 * static_cast<double>(allp.top1) / static_cast<double>(allp.n));
    chk.report(allp.nonfinite == 0, strf("no NaN/Inf logits (%zu positions)", allp.n));
    chk.report(allp.kl_mean() <= kTargetMeanKL,
               strf("mean KL(reference || quantized) %.4f <= %.2f nats per position (max %.4f)", allp.kl_mean(),
                    kTargetMeanKL, allp.kl_max));
    const double conf = allp.confident ? static_cast<double>(allp.confident_top1) / static_cast<double>(allp.confident) : 1.0;
    chk.report(conf >= kTargetConfidentTop1,
               strf("top-1 agreement where the reference is confident (p >= %.1f): %zu/%zu = %.1f%% >= %.0f%%",
                    kConfidentP, allp.confident_top1, allp.confident, 100.0 * conf, 100.0 * kTargetConfidentTop1));
  }
  if (R.ppl_ref > 0.0) {
    chk.report(R.ppl_q / R.ppl_ref <= kTargetPplRatio,
               strf("text perplexity ratio %.4f <= %.2f (%.3f -> %.3f)", R.ppl_q / R.ppl_ref, kTargetPplRatio,
                    R.ppl_ref, R.ppl_q));
  }
  if (R.greedy_prompts > 0) {
    std::printf("    greedy continuations identical to PyTorch: %zu/%zu prompts (reported, not a target)\n",
                R.greedy_identical, R.greedy_prompts);
  }
  std::printf("\nRESULT: %s  (%d passed, %d failed)\n", R.failed ? "FAIL" : "PASS", R.passed, R.failed);
  return R.failed ? 1 : 0;
}

}  // namespace ef
