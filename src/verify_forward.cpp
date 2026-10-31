// EdgeForge - verify-forward: full FP32 forward + KV cache vs PyTorch golden.
#include "ef/verify_forward.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

#include "ef/error.h"
#include "ef/format.h"
#include "ef/golden.h"
#include "ef/kernels.h"
#include "ef/model.h"

namespace fs = std::filesystem;

namespace ef {

namespace {

// FP32 engine vs FP32 PyTorch: different summation orders (BLAS vs our 8-lane
// dot) accumulate over 22 layers, so tolerances are looser than per-kernel ones,
// but a real bug (wrong RoPE pairing, GQA mapping, cache offset) gives cos << 0.99.
constexpr double kTolHidden = 1e-3, kCosHidden = 0.9999;
constexpr double kTolLogits = 1e-3, kCosLogits = 0.99999;
constexpr double kNearTie = 1e-3;  // golden top1-top2 margin below which a greedy flip is acceptable

struct Report {
  int pass = 0, fail = 0;
  bool row(const std::string& name, const DiffStats& d, double rel_tol, double cos_min) {
    const bool ok = d.non_finite == 0 && d.rel_err <= rel_tol && d.cosine >= cos_min;
    std::printf("  %-50s %10.3e %9.2e %12.9f  %s\n", name.c_str(), d.max_abs_err, d.rel_err, d.cosine,
                ok ? "PASS" : "FAIL");
    count(ok);
    return ok;
  }
  void flag(const std::string& name, bool ok, const std::string& detail) {
    std::printf("  %-50s %-34s  %s\n", name.c_str(), detail.c_str(), ok ? "PASS" : "FAIL");
    count(ok);
  }
  void count(bool ok) {
    if (ok) {
      ++pass;
    } else {
      ++fail;
    }
  }
};

// Keeps the worst-case statistics over many comparisons.
struct Worst {
  DiffStats d;
  bool any = false;
  void add(const DiffStats& s) {
    if (!any) {
      d = s;
      any = true;
      return;
    }
    d.max_abs_err = std::max(d.max_abs_err, s.max_abs_err);
    d.max_abs_ref = std::max(d.max_abs_ref, s.max_abs_ref);
    d.rel_err = std::max(d.rel_err, s.rel_err);
    d.cosine = std::min(d.cosine, s.cosine);
    d.non_finite += s.non_finite;
  }
};

std::string ids_preview(const std::vector<std::int32_t>& v, std::size_t n = 8) {
  std::string s;
  for (std::size_t i = 0; i < v.size() && i < n; ++i) s += (i ? " " : "") + std::to_string(v[i]);
  if (v.size() > n) s += " ...";
  return s;
}

void verify_prompt(const Model& m, const GoldenPrompt& g, const std::string& name, Report& rep) {
  const ModelConfig& c = m.config();
  const auto D = static_cast<std::size_t>(c.hidden_size);
  const auto V = static_cast<std::size_t>(c.vocab_size);
  const std::size_t L = m.n_layers();
  const std::vector<std::int32_t> ids = g.input_ids();
  const std::size_t S = ids.size();
  const json::Value& greedy = g.manifest().at("greedy");
  std::vector<std::int32_t> gold_ids;
  for (const auto& v : greedy.at("ids").items()) gold_ids.push_back(static_cast<std::int32_t>(v.as_int64()));
  const bool detail = g.has("hidden_all");
  std::printf("\n[%s] %zu prompt tokens%s; golden greedy: %zu tokens\n", name.c_str(), S,
              detail ? " (all-position detail)" : " (last-position data)", gold_ids.size());

  Transformer tr(m, S + gold_ids.size() + 1);
  ForwardTrace trace;
  std::vector<float> hidden_all, final_all, logits_all;  // ours, [L][S][D], [S][D], [S][V]
  if (detail) {
    hidden_all.resize(L * S * D);
    final_all.resize(S * D);
    logits_all.resize(S * V);
  }
  std::vector<float> hidden_last(L * D), final_last(D), logits_last(V);

  // ---------------- prefill
  const auto t0 = std::chrono::steady_clock::now();
  for (std::size_t p = 0; p < S; ++p) {
    const bool last = p + 1 == S;
    trace.capture = detail || last;
    const float* lg = tr.forward(ids[p], p, detail || last, &trace);
    if (detail) {
      for (std::size_t l = 0; l < L; ++l) {
        std::copy_n(trace.layer_out.data() + l * D, D, hidden_all.data() + (l * S + p) * D);
      }
      std::copy_n(trace.final_norm.data(), D, final_all.data() + p * D);
      std::copy_n(lg, V, logits_all.data() + p * V);
    }
    if (last) {
      hidden_last = trace.layer_out;
      final_last = trace.final_norm;
      std::copy_n(lg, V, logits_last.data());
    }
  }
  const double prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  // ---------------- hidden states
  if (detail) {
    const GoldenTensor gh = g.load("hidden_all");
    std::printf("  %-6s %10s %9s %12s\n", "layer", "max|err|", "rel err", "cosine");
    Worst all;
    for (std::size_t l = 0; l < L; ++l) {
      const DiffStats s = compare(hidden_all.data() + l * S * D, gh.f32.data() + l * S * D, S * D);
      all.add(s);
      std::printf("  %6zu %10.3e %9.2e %12.9f\n", l, s.max_abs_err, s.rel_err, s.cosine);
    }
    rep.row(strf("hidden states: %zu layers x %zu positions", L, S), all.d, kTolHidden, kCosHidden);
    rep.row("final RMSNorm, all positions", compare(final_all.data(), g.load("final_norm_all").f32.data(), S * D),
            kTolHidden, kCosHidden);
    const GoldenTensor gl = g.load("logits_all");
    rep.row(strf("logits, all %zu positions", S), compare(logits_all.data(), gl.f32.data(), S * V), kTolLogits,
            kCosLogits);
    std::size_t same = 0;
    for (std::size_t p = 0; p < S; ++p) {
      same += kern::argmax(logits_all.data() + p * V, V) == kern::argmax(gl.f32.data() + p * V, V);
    }
    rep.flag("next-token argmax at every position", same == S, strf("%zu/%zu identical", same, S));
  } else {
    const GoldenTensor gh = g.load("hidden_last");
    Worst all;
    for (std::size_t l = 0; l < L; ++l) all.add(compare(hidden_last.data() + l * D, gh.f32.data() + l * D, D));
    rep.row(strf("hidden states: %zu layers, last position", L), all.d, kTolHidden, kCosHidden);
  }
  rep.row("final RMSNorm, last position", compare(final_last.data(), g.load("final_norm_last").f32.data(), D),
          kTolHidden, kCosHidden);
  rep.row("next-token logits after the prompt", compare(logits_last.data(), g.load("logits_last").f32.data(), V),
          kTolLogits, kCosLogits);

  // ---------------- decode through the KV cache, greedy
  std::vector<float> gold_decode;
  std::size_t n_decode_gold = 0;
  if (g.has("decode_logits")) {
    const GoldenTensor gd = g.load("decode_logits");
    gold_decode = gd.f32;
    n_decode_gold = static_cast<std::size_t>(gd.shape.at(0));
  }
  const auto eos = static_cast<std::int32_t>(c.eos_token_id);
  std::vector<float> cur(logits_last);
  std::vector<std::int32_t> ours;
  Worst dec;
  std::size_t dec_top1 = 0, dec_n = 0, decode_forwards = 0, pos = S;
  bool diverged = false;
  double diverge_margin = 0.0;
  const auto t1 = std::chrono::steady_clock::now();
  for (std::size_t step = 0; step < gold_ids.size(); ++step) {
    const auto tok = static_cast<std::int32_t>(kern::argmax(cur.data(), V));
    ours.push_back(tok);
    if (tok != gold_ids[step]) {
      diverged = true;
      diverge_margin = greedy.at("steps")[step].at("margin_top1_top2").as_double();
      break;
    }
    if (tok == eos || step + 1 == gold_ids.size()) break;
    const float* lg = tr.forward(tok, pos++);
    ++decode_forwards;
    std::copy_n(lg, V, cur.data());
    if (step < n_decode_gold) {
      const float* ref = gold_decode.data() + step * V;
      dec.add(compare(cur.data(), ref, V));
      dec_top1 += kern::argmax(cur.data(), V) == kern::argmax(ref, V);
      ++dec_n;
    }
  }
  const double decode_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
  if (dec_n > 0) {
    rep.row(strf("decode logits via KV cache (%zu steps)", dec_n), dec.d, kTolLogits, kCosLogits);
    rep.flag("decode-step argmax", dec_top1 == dec_n, strf("%zu/%zu identical", dec_top1, dec_n));
  }
  const bool greedy_ok = !diverged || diverge_margin < kNearTie;
  rep.flag(strf("greedy tokens (%zu)", gold_ids.size()), greedy_ok,
           diverged ? strf("diverged at step %zu (HF margin %.4f)", ours.size() - 1, diverge_margin)
                    : strf("%zu/%zu identical", ours.size(), gold_ids.size()));
  std::printf("    ours: %s\n", ids_preview(ours, 12).c_str());

  const double bytes = static_cast<double>(m.weight_bytes_per_token());
  std::printf("  timing (scalar FP32, 1 thread): prefill %zu tok in %.2f s (%.2f tok/s%s)", S, prefill_s,
              static_cast<double>(S) / prefill_s, detail ? ", incl. lm_head at every position" : "");
  if (decode_forwards > 0) {
    const double tps = static_cast<double>(decode_forwards) / decode_s;
    std::printf("; decode %.2f tok/s = %.2f GB/s of weights", tps, tps * bytes / 1e9);
  }
  std::printf("\n");
}

}  // namespace

int run_verify_forward(const VerifyForwardOptions& opt) {
  std::string ref_desc;
  {
    std::ifstream in(opt.golden_dir / "meta.json", std::ios::binary);
    if (!in) throw Error("cannot open '" + (opt.golden_dir / "meta.json").string() + "'");
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const json::Value meta = json::parse(text);
    ref_desc = meta.at("reference_dtype").as_string() + ", transformers " +
               meta.at("versions").at("transformers").as_string();
  }
  const Model m = Model::load(opt.model_dir);
  const ModelConfig& c = m.config();
  std::printf("EdgeForge verify-forward: complete FP32 forward + KV cache vs PyTorch golden (%s)\n", ref_desc.c_str());
  std::printf("model: %lld layers, D %lld, %s of weights streamed per token\n",
              static_cast<long long>(c.num_hidden_layers), static_cast<long long>(c.hidden_size),
              human_bytes(m.weight_bytes_per_token()).c_str());
  std::printf("  %-50s %10s %9s %12s  %s\n", "check", "max|err|", "rel err", "cosine", "result");

  std::vector<std::string> prompts;
  if (!opt.prompt.empty()) {
    prompts.push_back(opt.prompt);
  } else {
    const fs::path root = opt.golden_dir / "prompts";
    if (!fs::is_directory(root)) throw Error("no prompts folder in '" + opt.golden_dir.string() + "'");
    for (const auto& e : fs::directory_iterator(root)) {
      if (e.is_directory() && fs::exists(e.path() / "manifest.json")) prompts.push_back(e.path().filename().string());
    }
    std::sort(prompts.begin(), prompts.end());
  }
  if (prompts.empty()) throw Error("no golden prompts found");

  Report rep;
  for (const auto& p : prompts) verify_prompt(m, GoldenPrompt::open(opt.golden_dir / "prompts" / p), p, rep);
  std::printf("\nRESULT: %s  (%d passed, %d failed; %zu prompts)\n", rep.fail ? "FAIL" : "PASS", rep.pass, rep.fail,
              prompts.size());
  return rep.fail ? 1 : 0;
}

}  // namespace ef
