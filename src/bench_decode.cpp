// EdgeForge - Stage 7: bench-decode.
#include "ef/bench_decode.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>

#include "ef/error.h"
#include "ef/format.h"
#include "ef/golden.h"
#include "ef/kernels.h"
#include "ef/model.h"
#include "ef/qkernels.h"
#include "ef/sysinfo.h"
#include "ef/threadpool.h"

namespace ef {

namespace {

struct RunOut {
  double prefill_s = 0.0, decode_s = 0.0;
  std::vector<std::int32_t> tokens;
  std::vector<float> logits;  // [tokens][V]: logits each decoded token was picked from
};

RunOut run_once(const Model& m, const std::vector<std::int32_t>& prompt, std::size_t n, ThreadPool* pool) {
  RunOut r;
  const auto V = static_cast<std::size_t>(m.config().vocab_size);
  Transformer tr(m, prompt.size() + n, pool);
  const auto t0 = std::chrono::steady_clock::now();
  const float* lg = nullptr;
  for (std::size_t p = 0; p < prompt.size(); ++p) lg = tr.forward(prompt[p], p, p + 1 == prompt.size());
  const auto t1 = std::chrono::steady_clock::now();
  r.prefill_s = std::chrono::duration<double>(t1 - t0).count();
  r.logits.reserve(n * V);
  for (std::size_t k = 0; k < n; ++k) {
    r.logits.insert(r.logits.end(), lg, lg + V);
    const auto tok = static_cast<std::int32_t>(kern::argmax(lg, V));
    r.tokens.push_back(tok);
    if (k + 1 < n) lg = tr.forward(tok, prompt.size() + k);
  }
  r.decode_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
  return r;
}

}  // namespace

int run_bench_decode(const BenchDecodeOptions& o) {
  if (o.qmodel.empty() == o.model_dir.empty()) throw Error("bench-decode needs exactly one of --qmodel / --model-dir");
  const Model m = o.qmodel.empty() ? Model::load(o.model_dir) : Model::load_efq(o.qmodel);
  const auto V = static_cast<std::size_t>(m.config().vocab_size);
  const std::vector<std::int32_t> prompt = o.prompt.empty() ? std::vector<std::int32_t>{1} : o.prompt;
  const std::size_t n = std::max<std::size_t>(o.tokens, 2);
  const SystemInfo si = query_system_info();
  const unsigned logical = std::max(1u, std::thread::hardware_concurrency());
  const unsigned cores = si.physical_cores > 0 ? static_cast<unsigned>(si.physical_cores) : logical;
  std::vector<unsigned> threads = o.threads;
  if (threads.empty()) threads = {2, 4, cores, logical};
  threads.erase(std::remove_if(threads.begin(), threads.end(), [](unsigned t) { return t < 2; }), threads.end());
  std::sort(threads.begin(), threads.end());
  threads.erase(std::unique(threads.begin(), threads.end()), threads.end());

  struct Config {
    kern::Backend b;
    unsigned t;
  };
  std::vector<Config> ladder = {{kern::Backend::Scalar, 1}};
  const kern::Backend fast = kern::avx2_compiled() ? kern::Backend::Avx2 : kern::Backend::Scalar;
  if (fast != kern::Backend::Scalar) ladder.push_back({fast, 1});
  for (unsigned t : threads) ladder.push_back({fast, t});
  if (fast != kern::Backend::Scalar && !threads.empty()) ladder.push_back({kern::Backend::Scalar, threads.back()});

  const double bytes = static_cast<double>(m.weight_bytes_per_token());
  std::printf("EdgeForge bench-decode: %s\n", m.description().c_str());
  std::printf("  %s | %d physical cores, %u logical | %s of weights per token\n", si.cpu_model.c_str(),
              si.physical_cores, logical, human_bytes(m.weight_bytes_per_token()).c_str());
  std::printf("  prompt %zu tokens, then %zu greedy decode steps per configuration (EOS ignored)\n",
              prompt.size(), n);

  const kern::Backend saved = kern::backend();
  {  // warm-up: fault in every weight page once, so the first configuration is not penalised
    kern::set_backend(fast);
    ThreadPool pool(threads.empty() ? 1 : threads.back());
    run_once(m, prompt, 2, &pool);
  }
  std::printf("\n  %-7s %7s %14s %13s %8s %7s %9s  %s\n", "kernels", "threads", "prefill tok/s", "decode tok/s",
              "GB/s", o.peak_gbps > 0 ? "% peak" : "", "speed-up", "outputs");
  std::vector<BenchRow> rows;
  RunOut scalar1, fast1;
  for (const Config& c : ladder) {
    kern::set_backend(c.b);
    std::unique_ptr<ThreadPool> pool = c.t > 1 ? std::make_unique<ThreadPool>(c.t) : nullptr;
    RunOut r = run_once(m, prompt, n, pool.get());
    BenchRow row;
    row.kernels = kern::backend_name(c.b);
    row.threads = c.t;
    row.prefill_tps = static_cast<double>(prompt.size()) / r.prefill_s;
    row.decode_tps = static_cast<double>(n - 1) / r.decode_s;  // n-1 forward passes in the decode loop
    row.gbps = row.decode_tps * bytes / 1e9;
    if (c.b == kern::Backend::Scalar && c.t == 1) scalar1 = r;
    if (c.b == fast && c.t == 1 && fast != kern::Backend::Scalar) fast1 = r;
    const RunOut& same1 = c.b == kern::Backend::Scalar ? scalar1 : fast1;
    row.bit_identical_to_1_thread = c.t == 1 || (r.tokens == same1.tokens && r.logits == same1.logits);
    while (row.tokens_same_as_scalar < n && r.tokens[row.tokens_same_as_scalar] == scalar1.tokens[row.tokens_same_as_scalar]) {
      ++row.tokens_same_as_scalar;
    }
    // Logits are comparable up to and including the first step whose token differs.
    const std::size_t comparable = std::min(n, row.tokens_same_as_scalar + 1);
    for (std::size_t k = 0; k < comparable; ++k) {
      const DiffStats d = compare(r.logits.data() + k * V, scalar1.logits.data() + k * V, V);
      row.max_rel_vs_scalar = std::max(row.max_rel_vs_scalar, d.rel_err);
    }
    std::string out;
    if (c.t > 1) out = row.bit_identical_to_1_thread ? "bit-identical to 1 thread" : "DIFFERS from 1 thread";
    if (c.b != kern::Backend::Scalar) {
      out += std::string(out.empty() ? "" : "; ") + strf("vs scalar: logits rel %.1e, tokens %zu/%zu", row.max_rel_vs_scalar,
                                                            row.tokens_same_as_scalar, n);
    }
    if (out.empty()) out = "reference";
    const double base = rows.empty() ? row.decode_tps : rows.front().decode_tps;
    std::printf("  %-7s %7u %14.2f %13.2f %8.2f %7s %8.2fx  %s\n", row.kernels, row.threads, row.prefill_tps,
                row.decode_tps, row.gbps, o.peak_gbps > 0 ? strf("%.0f%%", 100.0 * row.gbps / o.peak_gbps).c_str() : "",
                row.decode_tps / base, out.c_str());
    std::fflush(stdout);
    rows.push_back(row);
  }
  kern::set_backend(saved);
  bool ok = true;
  for (const auto& r : rows) ok = ok && r.bit_identical_to_1_thread;
  std::printf("\n  thread-count invariance: %s\n", ok ? "PASS (every thread count reproduces its 1-thread logits)" : "FAIL");
  if (o.peak_gbps > 0) std::printf("  membench peak: %.2f GB/s\n", o.peak_gbps);
  std::printf("  peak RSS %s\n", human_bytes(peak_rss_bytes()).c_str());
  if (o.rows != nullptr) *o.rows = rows;
  return ok ? 0 : 1;
}

}  // namespace ef
