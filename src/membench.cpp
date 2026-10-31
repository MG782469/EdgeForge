// EdgeForge — memory read bandwidth benchmark implementation.
#include "ef/membench.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <new>
#include <string>
#include <thread>

#include "ef/error.h"
#include "ef/sysinfo.h"

namespace ef {

namespace {

constexpr std::uint64_t kMul = 0x9E3779B97F4A7C15ull;
constexpr std::uint64_t kAdd = 0x0123456789ABCDEFull;
constexpr std::size_t kAlign = 4096;
constexpr std::size_t kWordsPerLine = 8;  // 64-byte cache line

inline std::uint64_t pattern(std::uint64_t i) noexcept { return i * kMul + kAdd; }

struct AlignedBuffer {
  std::uint64_t* p = nullptr;
  std::size_t words = 0;
  explicit AlignedBuffer(std::size_t w) : words(w) {
    p = static_cast<std::uint64_t*>(::operator new(w * sizeof(std::uint64_t), std::align_val_t{kAlign}, std::nothrow));
  }
  ~AlignedBuffer() {
    if (p != nullptr) ::operator delete(p, std::align_val_t{kAlign});
  }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
};

// Eight independent accumulators so the adds are not one serial chain; with
// -O3 the compiler turns this into wide vector loads + adds.
std::uint64_t sum_words(const std::uint64_t* p, std::size_t n) noexcept {
  std::uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    a0 += p[i + 0];
    a1 += p[i + 1];
    a2 += p[i + 2];
    a3 += p[i + 3];
    a4 += p[i + 4];
    a5 += p[i + 5];
    a6 += p[i + 6];
    a7 += p[i + 7];
  }
  for (; i < n; ++i) a0 += p[i];
  return ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
}

struct Range {
  std::size_t begin = 0;
  std::size_t end = 0;
};

// Splits `words` into `threads` contiguous, cache-line-aligned slices.
std::vector<Range> split(std::size_t words, int threads) {
  const std::size_t lines = words / kWordsPerLine;
  const auto t = static_cast<std::size_t>(threads);
  const std::size_t base = lines / t;
  const std::size_t extra = lines % t;
  std::vector<Range> r(t);
  std::size_t cursor = 0;
  for (std::size_t k = 0; k < t; ++k) {
    const std::size_t n = (base + (k < extra ? 1 : 0)) * kWordsPerLine;
    r[k] = {cursor, cursor + n};
    cursor += n;
  }
  r.back().end = words;  // tail words (< one line) go to the last thread
  return r;
}

struct alignas(64) Slot {
  std::uint64_t value = 0;
};

}  // namespace

std::uint64_t membench_expected_pass_sum(std::uint64_t n) noexcept {
  // sum_{i<n} (i*kMul + kAdd) = kMul * n(n-1)/2 + kAdd * n   (mod 2^64)
  const std::uint64_t tri = (n % 2 == 0) ? (n / 2) * (n - 1) : n * ((n - 1) / 2);
  return kMul * tri + kAdd * n;
}

double median_of(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

std::vector<MembenchResult> run_membench(const MembenchOptions& opt,
                                         const std::function<void(const MembenchResult&)>& on_result) {
  if (opt.buffer_bytes < 64 * 1024) throw Error("membench: buffer must be at least 64 KiB");
  if (opt.thread_counts.empty()) throw Error("membench: no thread counts given");
  for (int t : opt.thread_counts) {
    if (t < 1 || t > 1024) throw Error("membench: thread count " + std::to_string(t) + " is outside [1, 1024]");
  }
  if (opt.runs < 1) throw Error("membench: --runs must be >= 1");
  if (opt.passes < 1) throw Error("membench: --passes must be >= 1");
  if (opt.warmup_runs < 0) throw Error("membench: --warmup must be >= 0");

  const std::size_t words = static_cast<std::size_t>(opt.buffer_bytes / sizeof(std::uint64_t));
  const int max_threads = *std::max_element(opt.thread_counts.begin(), opt.thread_counts.end());
  if (words / kWordsPerLine < static_cast<std::size_t>(max_threads)) {
    throw Error("membench: buffer too small for " + std::to_string(max_threads) + " threads");
  }

  const SystemInfo sys = query_system_info();
  if (!opt.allow_low_memory && sys.avail_ram_bytes > 0 &&
      static_cast<double>(opt.buffer_bytes) > 0.8 * static_cast<double>(sys.avail_ram_bytes)) {
    throw Error("membench: " + std::to_string(opt.buffer_bytes >> 20) + " MiB requested but only " +
                std::to_string(sys.avail_ram_bytes >> 20) +
                " MiB of RAM is available; use a smaller --mb or pass --force");
  }

  AlignedBuffer buf(words);
  if (buf.p == nullptr) {
    throw Error("membench: could not allocate " + std::to_string(opt.buffer_bytes >> 20) + " MiB");
  }

  // Parallel first touch: every page is written (and therefore backed by RAM)
  // before any timing starts.
  {
    const auto ranges = split(words, max_threads);
    std::vector<std::thread> pool;
    for (const Range& r : ranges) {
      pool.emplace_back([&buf, r] {
        for (std::size_t i = r.begin; i < r.end; ++i) buf.p[i] = pattern(i);
      });
    }
    for (auto& th : pool) th.join();
  }

  const std::uint64_t pass_sum = membench_expected_pass_sum(words);
  const std::uint64_t bytes_per_pass = static_cast<std::uint64_t>(words) * sizeof(std::uint64_t);
  std::vector<MembenchResult> results;

  for (const int T : opt.thread_counts) {
    const auto ranges = split(words, T);
    const int total_runs = opt.warmup_runs + opt.runs;
    std::vector<Slot> slots(static_cast<std::size_t>(T));
    std::vector<double> seconds(static_cast<std::size_t>(total_runs), 0.0);
    std::vector<std::uint64_t> sums(static_cast<std::size_t>(total_runs), 0);

    struct PhaseState {
      std::chrono::steady_clock::time_point t0{};
      int phase = 0;
      std::vector<double>* seconds = nullptr;
      std::vector<std::uint64_t>* sums = nullptr;
      std::vector<Slot>* slots = nullptr;
    } st;
    st.seconds = &seconds;
    st.sums = &sums;
    st.slots = &slots;

    // Runs once per barrier phase, after every worker has arrived.
    auto on_phase = [&st]() noexcept {
      const auto now = std::chrono::steady_clock::now();
      if (st.phase % 2 == 0) {
        st.t0 = now;  // all workers ready: start the clock
      } else {
        const auto run = static_cast<std::size_t>(st.phase / 2);
        (*st.seconds)[run] = std::chrono::duration<double>(now - st.t0).count();
        std::uint64_t total = 0;
        for (const Slot& s : *st.slots) total += s.value;
        (*st.sums)[run] = total;
      }
      ++st.phase;
    };
    std::barrier sync(T, on_phase);

    std::vector<std::thread> pool;
    for (int t = 0; t < T; ++t) {
      pool.emplace_back([&, t] {
        const Range r = ranges[static_cast<std::size_t>(t)];
        const std::uint64_t* base = buf.p + r.begin;
        const std::size_t n = r.end - r.begin;
        for (int run = 0; run < total_runs; ++run) {
          sync.arrive_and_wait();  // start of timed region
          std::uint64_t acc = 0;
          for (int p = 0; p < opt.passes; ++p) {
            acc += sum_words(base, n);
            // Compiler barrier: each pass must really re-read memory.
            std::atomic_signal_fence(std::memory_order_seq_cst);
          }
          slots[static_cast<std::size_t>(t)].value = acc;
          sync.arrive_and_wait();  // end of timed region
        }
      });
    }
    for (auto& th : pool) th.join();

    MembenchResult res;
    res.threads = T;
    res.buffer_bytes = bytes_per_pass;
    res.passes = opt.passes;
    res.bytes_per_run = bytes_per_pass * static_cast<std::uint64_t>(opt.passes);
    res.expected_checksum = pass_sum * static_cast<std::uint64_t>(opt.passes);
    res.checksum_ok = true;
    for (int run = 0; run < total_runs; ++run) {
      const auto k = static_cast<std::size_t>(run);
      if (sums[k] != res.expected_checksum) res.checksum_ok = false;
      if (run < opt.warmup_runs) continue;
      res.run_seconds.push_back(seconds[k]);
      res.run_gbps.push_back(seconds[k] > 0 ? static_cast<double>(res.bytes_per_run) / seconds[k] / 1e9 : 0.0);
    }
    res.checksum = sums.back();
    res.total_bytes_read = res.bytes_per_run * static_cast<std::uint64_t>(total_runs);
    res.best_gbps = *std::max_element(res.run_gbps.begin(), res.run_gbps.end());
    res.worst_gbps = *std::min_element(res.run_gbps.begin(), res.run_gbps.end());
    res.median_gbps = median_of(res.run_gbps);
    if (on_result) on_result(res);
    results.push_back(std::move(res));
  }
  return results;
}

}  // namespace ef
