// EdgeForge — membench correctness tests (checksums across thread counts).
#include <cstdint>

#include "ef/membench.h"
#include "test_framework.h"

using namespace ef;

TEST(membench_closed_form_checksum) {
  // The closed form must equal a brute-force sum of the fill pattern.
  for (std::uint64_t n : {0ull, 1ull, 2ull, 3ull, 7ull, 8ull, 1000ull, 1001ull, 65537ull}) {
    std::uint64_t brute = 0;
    for (std::uint64_t i = 0; i < n; ++i) brute += i * 0x9E3779B97F4A7C15ull + 0x0123456789ABCDEFull;
    CHECK_EQ(membench_expected_pass_sum(n), brute);
  }
}

TEST(membench_checksums_match_for_all_thread_counts) {
  MembenchOptions o;
  o.buffer_bytes = (4u << 20) + 24;  // not a multiple of a cache line: exercises the tail
  o.thread_counts = {1, 2, 3, 4, 7};
  o.runs = 2;
  o.passes = 2;
  o.warmup_runs = 1;
  const auto results = run_membench(o);
  CHECK_EQ(results.size(), std::size_t{5});
  const std::uint64_t words = o.buffer_bytes / 8;
  for (const auto& r : results) {
    CHECK(r.checksum_ok);
    CHECK_EQ(r.checksum, r.expected_checksum);
    CHECK_EQ(r.expected_checksum, membench_expected_pass_sum(words) * 2);
    CHECK_EQ(r.buffer_bytes, words * 8);
    CHECK_EQ(r.bytes_per_run, words * 8 * 2);
    CHECK_EQ(r.total_bytes_read, words * 8 * 2 * 3);
    CHECK_EQ(r.run_gbps.size(), std::size_t{2});
    CHECK(r.worst_gbps > 0.0 && r.worst_gbps <= r.median_gbps && r.median_gbps <= r.best_gbps);
  }
}

TEST(membench_rejects_bad_options) {
  MembenchOptions o;
  o.buffer_bytes = 1000;
  CHECK_THROWS(run_membench(o), "at least 64 KiB");
  o = MembenchOptions{};
  o.buffer_bytes = 1 << 20;
  o.thread_counts = {0};
  CHECK_THROWS(run_membench(o), "outside [1, 1024]");
  o.thread_counts = {};
  CHECK_THROWS(run_membench(o), "no thread counts");
  o.thread_counts = {1};
  o.runs = 0;
  CHECK_THROWS(run_membench(o), "--runs");
  o.runs = 1;
  o.passes = 0;
  CHECK_THROWS(run_membench(o), "--passes");
  o.passes = 1;
  o.warmup_runs = -1;
  CHECK_THROWS(run_membench(o), "--warmup");
  o.warmup_runs = 0;
  o.buffer_bytes = 1ull << 50;  // 1 PiB: refused by the RAM guard or the allocator, never crashes
  CHECK_THROWS(run_membench(o), "MiB");
}

TEST(membench_median) {
  CHECK_EQ(median_of({3.0, 1.0, 2.0}), 2.0);
  CHECK_EQ(median_of({4.0, 1.0, 2.0, 3.0}), 2.5);
  CHECK_EQ(median_of({}), 0.0);
}
