// EdgeForge - Stage 7: `edgeforge bench-decode`, the speed-up ladder.
//
// Runs the same prompt + greedy decode with each configuration
//     scalar / 1 thread  ->  avx2 / 1 thread  ->  avx2 / 2, 4, ... threads  (+ scalar / max threads)
// and reports prefill and decode tok/s, GB/s of weights streamed, % of the membench baseline and the
// speed-up over scalar / 1 thread. It also checks the outputs: every thread count of a backend
// must give bit-identical logits, and the AVX2 logits are compared with scalar ones.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace ef {

struct BenchRow {
  const char* kernels = "";
  unsigned threads = 1;
  double prefill_tps = 0.0, decode_tps = 0.0, gbps = 0.0;
  bool bit_identical_to_1_thread = true;  // vs the same backend with 1 thread
  std::size_t tokens_same_as_scalar = 0;  // greedy tokens equal to scalar/1 (prefix length)
  double max_rel_vs_scalar = 0.0;         // logits, over the steps before any token differs
};

struct BenchDecodeOptions {
  std::filesystem::path qmodel;     // .efq (quantized engine) ...
  std::filesystem::path model_dir;  // ... or the BF16 checkpoint
  std::vector<std::int32_t> prompt;
  std::size_t tokens = 32;          // decoded tokens per configuration (EOS does not stop it)
  std::vector<unsigned> threads;    // thread counts for the SIMD backend (empty: 2, 4, cores, logical)
  double peak_gbps = 0.0;           // membench baseline for the "% mbench" column (0 = unknown)
  bool ladder = true;               // false: only the SIMD backend at `threads` (no scalar / 1-thread rows)
  std::vector<BenchRow>* rows = nullptr;  // optional: results (tests)
};

int run_bench_decode(const BenchDecodeOptions& o);

// Stage 8: `edgeforge profile` - per-op time breakdown of decoding (prompt, then `tokens` greedy
// steps; only the decode steps are timed), with GB/s of weights for each group of matrices.
class ThreadPool;
struct ProfileOptions {
  std::filesystem::path qmodel, model_dir;  // exactly one
  std::vector<std::int32_t> prompt;
  std::size_t tokens = 32;
  ThreadPool* pool = nullptr;
  double peak_gbps = 0.0;  // membench baseline (0 = unknown)
};
int run_profile(const ProfileOptions& o);

}  // namespace ef
