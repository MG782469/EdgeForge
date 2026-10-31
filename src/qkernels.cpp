// EdgeForge - Stage 6: packed Q4_0 / Q8_0 GEMV kernels; Stage 7: AVX2 backend + row-parallel GEMV.
#include "ef/qkernels.h"

#include <algorithm>
#include <atomic>
#include <cstring>

#include "ef/error.h"
#include "ef/fp16.h"
#include "ef/kernels.h"
#include "ef/threadpool.h"

#if defined(EF_HAVE_AVX2)
#include <immintrin.h>
#endif

namespace ef::kern {

namespace {

inline float load_scale(const std::uint8_t* p) noexcept {
  std::uint16_t h;
  std::memcpy(&h, p, 2);
  return fp16_to_f32(h);
}

// 32 FP32 lanes -> one value in a fixed order, so results do not depend on the compiler.
inline float sum32(const float* a, const float* b) noexcept {
  float s[8];
  for (std::size_t j = 0; j < 8; ++j) s[j] = (a[j] + a[j + 8]) + (b[j] + b[j + 8]);
  return ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
}

}  // namespace

// 32 FP32 lane accumulators run across the whole row (lane j holds weights j of every block),
// so a block costs one vectorisable 16-wide loop and no horizontal sum. Weights j and j+16 share
// byte j (low / high nibble). Written so GCC/Clang auto-vectorise it; explicit SIMD is Stage 7.
float dot_q4_0(const std::uint8_t* row, const float* x, std::size_t n) noexcept {
  constexpr std::size_t kBlock = sizeof(BlockQ4_0);  // 18
  float lo[16] = {}, hi[16] = {};
  for (std::size_t b = 0; b < n / kQK; ++b, row += kBlock, x += kQK) {
    const float d = load_scale(row);
    const std::uint8_t* qs = row + 2;
    for (std::size_t j = 0; j < 16; ++j) {
      lo[j] += d * (static_cast<float>((qs[j] & 0x0F) - 8) * x[j]);
      hi[j] += d * (static_cast<float>((qs[j] >> 4) - 8) * x[j + 16]);
    }
  }
  return sum32(lo, hi);
}

float dot_q8_0(const std::uint8_t* row, const float* x, std::size_t n) noexcept {
  constexpr std::size_t kBlock = sizeof(BlockQ8_0);  // 34
  float lo[16] = {}, hi[16] = {};
  for (std::size_t b = 0; b < n / kQK; ++b, row += kBlock, x += kQK) {
    const float d = load_scale(row);
    const auto* qs = reinterpret_cast<const std::int8_t*>(row + 2);
    for (std::size_t j = 0; j < 16; ++j) {
      lo[j] += d * (static_cast<float>(qs[j]) * x[j]);
      hi[j] += d * (static_cast<float>(qs[j + 16]) * x[j + 16]);
    }
  }
  return sum32(lo, hi);
}

void matvec_q4_0(float* y, const std::uint8_t* W, const float* x, std::size_t rows, std::size_t cols) noexcept {
  const std::size_t rb = cols / kQK * sizeof(BlockQ4_0);
  for (std::size_t r = 0; r < rows; ++r) y[r] = dot_q4_0(W + r * rb, x, cols);
}

void matvec_q8_0(float* y, const std::uint8_t* W, const float* x, std::size_t rows, std::size_t cols) noexcept {
  const std::size_t rb = cols / kQK * sizeof(BlockQ8_0);
  for (std::size_t r = 0; r < rows; ++r) y[r] = dot_q8_0(W + r * rb, x, cols);
}

// ------------------------------------------------------------------ backend

namespace {
std::atomic<Backend> g_backend{default_backend()};
}  // namespace

bool avx2_compiled() noexcept {
#if defined(EF_HAVE_AVX2)
  return true;
#else
  return false;
#endif
}

Backend default_backend() noexcept { return avx2_compiled() ? Backend::Avx2 : Backend::Scalar; }
Backend backend() noexcept { return g_backend.load(std::memory_order_relaxed); }

void set_backend(Backend b) {
  if (b == Backend::Avx2 && !avx2_compiled()) {
    throw Error("this build has no AVX2 kernels (compile with -march=native / EF_NATIVE=ON on an AVX2 CPU)");
  }
  g_backend.store(b, std::memory_order_relaxed);
}

const char* backend_name(Backend b) noexcept { return b == Backend::Avx2 ? "avx2" : "scalar"; }

Backend backend_from_string(const std::string& s) {
  if (s == "auto") return default_backend();
  if (s == "scalar") return Backend::Scalar;
  if (s == "avx2") return Backend::Avx2;
  throw Error("--kernels expects scalar, avx2 or auto; got '" + s + "'");
}

// ------------------------------------------------------------------ AVX2 + FMA + F16C kernels

#if defined(EF_HAVE_AVX2)
namespace {

inline float hsum(__m256 v) noexcept {
  __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  a = _mm_add_ps(a, _mm_movehl_ps(a, a));
  a = _mm_add_ss(a, _mm_movehdup_ps(a));
  return _mm_cvtss_f32(a);
}

inline float scale_of(const std::uint8_t* blk) noexcept {
  std::uint16_t h;
  std::memcpy(&h, blk, 2);
  return _cvtsh_ss(h);  // exact FP16 -> FP32 in hardware
}

inline __m256 i8x8_to_ps(__m128i v) noexcept { return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v)); }

// acc += d * sum_j w_j x_j for one Q4_0 block (codes widened to signed bytes, then to floats).
inline __m256 q4_block(const std::uint8_t* blk, const float* x, __m256 acc) noexcept {
  const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
  const __m128i m = _mm_set1_epi8(0x0F), eight = _mm_set1_epi8(8);
  const __m128i lo = _mm_sub_epi8(_mm_and_si128(q, m), eight);                      // weights 0..15
  const __m128i hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(q, 4), m), eight);  // weights 16..31
  __m256 s = _mm256_mul_ps(i8x8_to_ps(lo), _mm256_loadu_ps(x));
  s = _mm256_fmadd_ps(i8x8_to_ps(_mm_srli_si128(lo, 8)), _mm256_loadu_ps(x + 8), s);
  s = _mm256_fmadd_ps(i8x8_to_ps(hi), _mm256_loadu_ps(x + 16), s);
  s = _mm256_fmadd_ps(i8x8_to_ps(_mm_srli_si128(hi, 8)), _mm256_loadu_ps(x + 24), s);
  return _mm256_fmadd_ps(_mm256_set1_ps(scale_of(blk)), s, acc);
}

inline __m256 q8_block(const std::uint8_t* blk, const float* x, __m256 acc) noexcept {
  const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
  const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 18));
  __m256 s = _mm256_mul_ps(i8x8_to_ps(a), _mm256_loadu_ps(x));
  s = _mm256_fmadd_ps(i8x8_to_ps(_mm_srli_si128(a, 8)), _mm256_loadu_ps(x + 8), s);
  s = _mm256_fmadd_ps(i8x8_to_ps(b), _mm256_loadu_ps(x + 16), s);
  s = _mm256_fmadd_ps(i8x8_to_ps(_mm_srli_si128(b, 8)), _mm256_loadu_ps(x + 24), s);
  return _mm256_fmadd_ps(_mm256_set1_ps(scale_of(blk)), s, acc);
}

}  // namespace

// Two independent accumulators (even / odd blocks) hide the FMA latency.
float dot_q4_0_avx2(const std::uint8_t* row, const float* x, std::size_t n) noexcept {
  constexpr std::size_t kB = sizeof(BlockQ4_0);
  const std::size_t nb = n / kQK;
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  std::size_t b = 0;
  for (; b + 2 <= nb; b += 2) {
    a0 = q4_block(row + b * kB, x + b * kQK, a0);
    a1 = q4_block(row + (b + 1) * kB, x + (b + 1) * kQK, a1);
  }
  if (b < nb) a0 = q4_block(row + b * kB, x + b * kQK, a0);
  return hsum(_mm256_add_ps(a0, a1));
}

float dot_q8_0_avx2(const std::uint8_t* row, const float* x, std::size_t n) noexcept {
  constexpr std::size_t kB = sizeof(BlockQ8_0);
  const std::size_t nb = n / kQK;
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  std::size_t b = 0;
  for (; b + 2 <= nb; b += 2) {
    a0 = q8_block(row + b * kB, x + b * kQK, a0);
    a1 = q8_block(row + (b + 1) * kB, x + (b + 1) * kQK, a1);
  }
  if (b < nb) a0 = q8_block(row + b * kB, x + b * kQK, a0);
  return hsum(_mm256_add_ps(a0, a1));
}

float dot_bf16_avx2(const std::uint16_t* w, const float* x, std::size_t n) noexcept {
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m256i u0 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i)));
    const __m256i u1 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i + 8)));
    a0 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(u0, 16)), _mm256_loadu_ps(x + i), a0);
    a1 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(u1, 16)), _mm256_loadu_ps(x + i + 8), a1);
  }
  float s = hsum(_mm256_add_ps(a0, a1));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * x[i];
  return s;
}

float dot_f32_avx2(const float* w, const float* x, std::size_t n) noexcept {
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(x + i), a0);
    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 8), _mm256_loadu_ps(x + i + 8), a1);
    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 16), _mm256_loadu_ps(x + i + 16), a2);
    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 24), _mm256_loadu_ps(x + i + 24), a3);
  }
  float s = hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
  for (; i < n; ++i) s += w[i] * x[i];
  return s;
}
#endif

// ------------------------------------------------------------------ dispatch

void matvec_rows(float* y, const WeightView& W, const float* x, std::size_t begin, std::size_t end) noexcept {
  if (begin >= end) return;
  const std::size_t rb = W.row_bytes();
  const std::uint8_t* base = W.data + begin * rb;
  float* out = y + begin;
  const std::size_t rows = end - begin;
#if defined(EF_HAVE_AVX2)
  if (backend() == Backend::Avx2) {
    for (std::size_t r = 0; r < rows; ++r) {
      const std::uint8_t* row = base + r * rb;
      switch (W.type) {
        case QType::Q4_0: out[r] = dot_q4_0_avx2(row, x, W.cols); break;
        case QType::Q8_0: out[r] = dot_q8_0_avx2(row, x, W.cols); break;
        case QType::BF16: out[r] = dot_bf16_avx2(reinterpret_cast<const std::uint16_t*>(row), x, W.cols); break;
        case QType::F32: out[r] = dot_f32_avx2(reinterpret_cast<const float*>(row), x, W.cols); break;
        case QType::F16: {
          float acc = 0.0f;
          for (std::size_t c = 0; c < W.cols; ++c) {
            std::uint16_t h;
            std::memcpy(&h, row + 2 * c, 2);
            acc += fp16_to_f32(h) * x[c];
          }
          out[r] = acc;
          break;
        }
      }
    }
    return;
  }
#endif
  switch (W.type) {
    case QType::BF16:
      matvec_bf16(out, reinterpret_cast<const std::uint16_t*>(base), x, rows, W.cols);
      return;
    case QType::F32:
      matvec_f32(out, reinterpret_cast<const float*>(base), x, rows, W.cols);
      return;
    case QType::Q4_0:
      matvec_q4_0(out, base, x, rows, W.cols);
      return;
    case QType::Q8_0:
      matvec_q8_0(out, base, x, rows, W.cols);
      return;
    case QType::F16:
      for (std::size_t r = 0; r < rows; ++r) {
        const std::uint8_t* p = base + r * rb;
        float acc = 0.0f;
        for (std::size_t c = 0; c < W.cols; ++c) {
          std::uint16_t h;
          std::memcpy(&h, p + 2 * c, 2);
          acc += fp16_to_f32(h) * x[c];
        }
        out[r] = acc;
      }
      return;
  }
}

namespace {
constexpr std::size_t kMinRowsPerThread = 32;  // below this, waking another thread costs more than it saves
}  // namespace

void matvec(float* y, const WeightView& W, const float* x, ThreadPool* pool) noexcept {
  const MatvecJob job{y, &W, x};
  matvec_jobs(&job, 1, pool);
}

void matvec_jobs(const MatvecJob* jobs, std::size_t n_jobs, ThreadPool* pool) noexcept {
  std::size_t total = 0;
  for (std::size_t j = 0; j < n_jobs; ++j) total += jobs[j].W->rows;
  // Rows of all jobs form one index space [0, total); each chunk maps back onto the jobs.
  auto body = [&](std::size_t, std::size_t b, std::size_t e) {
    std::size_t first = 0;
    for (std::size_t j = 0; j < n_jobs; ++j) {
      const std::size_t rows = jobs[j].W->rows, lo = std::max(b, first), hi = std::min(e, first + rows);
      if (lo < hi) matvec_rows(jobs[j].y, *jobs[j].W, jobs[j].x, lo - first, hi - first);
      first += rows;
    }
  };
  if (pool == nullptr || pool->size() <= 1) {
    body(0, 0, total);
  } else {
    pool->parallel_for(total, body, kMinRowsPerThread);
  }
}

void row_f32(float* out, const WeightView& W, std::size_t r) noexcept {
  if (W.type == QType::BF16) {  // same as the Stage 2 embedding kernel
    embedding_bf16(out, reinterpret_cast<const std::uint16_t*>(W.data), r, W.cols);
    return;
  }
  dequantize_row(W.type, W.data + r * W.row_bytes(), W.cols, out);
}

}  // namespace ef::kern
