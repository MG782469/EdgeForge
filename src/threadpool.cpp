// EdgeForge - Stage 7: persistent thread pool.
#include "ef/threadpool.h"

#include <algorithm>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#define EF_CPU_RELAX() _mm_pause()
#else
#define EF_CPU_RELAX() std::this_thread::yield()
#endif

namespace ef {

namespace {
// ~0.1-0.5 ms of polling after each job before a worker sleeps: covers the gaps between the
// matrix multiplies of one token (norms, RoPE, attention) without keeping cores busy when idle.
constexpr int kSpinIterations = 20000;
}  // namespace

ThreadPool::ThreadPool(unsigned threads) {
  size_ = threads != 0 ? threads : std::max(1u, std::thread::hardware_concurrency());
  for (unsigned i = 1; i < size_; ++i) threads_.emplace_back([this, i] { worker(i); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_.store(true, std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);
  }
  cv_.notify_all();
  for (auto& t : threads_) t.join();
}

void ThreadPool::run(std::size_t n, unsigned chunks, Call call, void* ctx) {
  job_n_ = n;
  job_chunks_ = chunks;
  job_call_ = call;
  job_ctx_ = ctx;
  // Every worker acknowledges every job (even with no chunk to run), so none can still be
  // reading the job fields above when the next job overwrites them.
  pending_.store(size_ - 1, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lk(mu_);
    generation_.fetch_add(1, std::memory_order_release);
  }
  cv_.notify_all();
  call(ctx, 0, 0, chunk_begin(n, chunks, 1));
  while (pending_.load(std::memory_order_acquire) != 0) EF_CPU_RELAX();
}

void ThreadPool::worker(unsigned id) {
  std::uint64_t seen = 0;
  for (;;) {
    std::uint64_t g = generation_.load(std::memory_order_acquire);
    for (int i = 0; g == seen && i < kSpinIterations; ++i) {
      EF_CPU_RELAX();
      g = generation_.load(std::memory_order_acquire);
    }
    if (g == seen) {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&] { return generation_.load(std::memory_order_acquire) != seen; });
      g = generation_.load(std::memory_order_acquire);
    }
    if (stop_.load(std::memory_order_relaxed)) return;  // set before the generation bump we just saw
    seen = g;
    if (id < job_chunks_) {
      job_call_(job_ctx_, id, chunk_begin(job_n_, job_chunks_, id), chunk_begin(job_n_, job_chunks_, id + 1));
    }
    pending_.fetch_sub(1, std::memory_order_release);
  }
}

}  // namespace ef
