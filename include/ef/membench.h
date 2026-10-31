// EdgeForge — STREAM-style multithreaded memory *read* bandwidth benchmark.
//
// Each worker sums 64-bit words over its own contiguous, cache-line-aligned
// slice of one large buffer (default 1 GiB, far larger than any laptop L3).
// The sum is checked against a closed-form expected value, so the reads can
// neither be skipped by the compiler nor silently wrong. Timing uses
// std::chrono::steady_clock between two std::barrier phases, so thread
// start-up is excluded.
//
// This measures achievable read bandwidth from software. It is NOT a hardware
// DRAM counter and will be below the RAM's theoretical (spec-sheet) bandwidth.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace ef {

struct MembenchOptions {
  std::uint64_t buffer_bytes = 1ull << 30;  // working set
  std::vector<int> thread_counts = {1};
  int runs = 5;          // timed runs per thread count
  int passes = 3;        // full sweeps of the buffer per run
  int warmup_runs = 1;   // untimed runs per thread count
  bool allow_low_memory = false;  // skip the free-RAM guard
};

struct MembenchResult {
  int threads = 0;
  std::uint64_t buffer_bytes = 0;   // bytes swept per pass
  int passes = 0;
  std::uint64_t bytes_per_run = 0;  // buffer_bytes * passes
  std::vector<double> run_seconds;
  std::vector<double> run_gbps;     // decimal GB/s (1e9 bytes)
  double best_gbps = 0.0;
  double median_gbps = 0.0;
  double worst_gbps = 0.0;
  std::uint64_t total_bytes_read = 0;  // timed + warm-up
  std::uint64_t checksum = 0;          // from the last timed run
  std::uint64_t expected_checksum = 0;
  bool checksum_ok = false;            // every run (incl. warm-up) matched
};

// Runs the benchmark for every thread count in options.thread_counts, reusing
// one buffer. on_result (optional) is called after each thread count finishes.
std::vector<MembenchResult> run_membench(const MembenchOptions& options,
                                         const std::function<void(const MembenchResult&)>& on_result = {});

// Expected sum of one full pass over `words` pattern words (mod 2^64).
std::uint64_t membench_expected_pass_sum(std::uint64_t words) noexcept;

double median_of(std::vector<double> v);

}  // namespace ef
