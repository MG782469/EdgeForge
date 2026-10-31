// EdgeForge - Stage 7: a small persistent thread pool for row-parallel GEMV.
//
// parallel_for(n, fn) splits [0, n) into contiguous chunks, one per thread; the calling thread
// runs chunk 0 and the workers the rest. Chunk boundaries depend only on n and the chunk count,
// and every output row is computed entirely by one thread with the same kernel, so results are
// bit-identical for any thread count.
//
// Decoding calls parallel_for ~90 times per token, so waking the workers must be cheap: they
// spin briefly on an atomic generation counter after each job and only then sleep on a
// condition variable (an idle chat does not burn CPU).
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <algorithm>
#include <thread>
#include <type_traits>
#include <vector>

namespace ef {

class ThreadPool {
 public:
  // threads = total threads including the caller (0 = std::thread::hardware_concurrency()).
  explicit ThreadPool(unsigned threads = 0);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  unsigned size() const noexcept { return size_; }

  // fn(chunk, begin, end) for chunk = 0..C-1 covering [0, n), C = min(size, ceil(n / min_chunk)).
  // Blocks until every chunk is done. fn must not throw. Not reentrant (one job at a time).
  template <class Fn>
  void parallel_for(std::size_t n, Fn&& fn, std::size_t min_chunk = 1) {
    if (n == 0) return;
    const std::size_t want = (n + min_chunk - 1) / (min_chunk == 0 ? 1 : min_chunk);
    const auto chunks = static_cast<unsigned>(std::min<std::size_t>(size_, std::max<std::size_t>(want, 1)));
    if (chunks <= 1) {
      fn(std::size_t{0}, std::size_t{0}, n);
      return;
    }
    using F = std::remove_reference_t<Fn>;
    run(n, chunks, [](void* ctx, std::size_t c, std::size_t b, std::size_t e) { (*static_cast<F*>(ctx))(c, b, e); },
        const_cast<void*>(static_cast<const void*>(&fn)));
  }

  // [begin, end) of chunk c out of `chunks` for n items (the split parallel_for uses).
  static std::size_t chunk_begin(std::size_t n, unsigned chunks, unsigned c) noexcept {
    return static_cast<std::size_t>(static_cast<std::uint64_t>(n) * c / chunks);  // n * c fits easily
  }

 private:
  using Call = void (*)(void*, std::size_t, std::size_t, std::size_t);
  void run(std::size_t n, unsigned chunks, Call call, void* ctx);
  void worker(unsigned id);

  unsigned size_ = 1;
  std::vector<std::thread> threads_;
  // Current job (written by the caller before `generation_` is bumped, read by workers after).
  std::size_t job_n_ = 0;
  unsigned job_chunks_ = 0;
  Call job_call_ = nullptr;
  void* job_ctx_ = nullptr;
  std::atomic<std::uint64_t> generation_{0};
  std::atomic<unsigned> pending_{0};  // workers that have not finished the current job
  std::mutex mu_;
  std::condition_variable cv_;
  std::atomic<bool> stop_{false};
};

}  // namespace ef
