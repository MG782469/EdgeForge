// EdgeForge - Stage 6: GEMV straight from the packed .efq weights.
//
// The quantized engine never builds a dequantized copy of a matrix: every dot product reads the
// 18-byte Q4_0 / 34-byte Q8_0 blocks in place (memory-mapped) and expands one block at a time
// into registers. Per block of 32 weights:
//     Q4_0:  d * sum_j (code_j - 8) * x_j        Q8_0:  d * sum_j q_j * x_j
// so the scale multiplies once per block instead of once per weight. FP32 activations and
// accumulation.
//
// Stage 7: two backends. "scalar" is the portable reference (Stage 6 numerics, auto-vectorised by
// the compiler); "avx2" uses explicit AVX2 + FMA + F16C intrinsics (8-wide FMA; different
// rounding order, so results agree with scalar to ~1e-7 relative, not bit for bit). Row-parallel
// execution over a ThreadPool computes each output row entirely in one thread, so for a given
// backend the results are bit-identical for any thread count.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "ef/quant.h"

namespace ef {

class ThreadPool;

// A row-major [rows][cols] weight matrix in any storage type, viewed in place (zero-copy).
struct WeightView {
  QType type = QType::BF16;
  const std::uint8_t* data = nullptr;
  std::size_t rows = 0, cols = 0;
  std::size_t row_bytes() const { return static_cast<std::size_t>(qtype_row_bytes(type, cols)); }
  std::uint64_t bytes() const { return static_cast<std::uint64_t>(row_bytes()) * rows; }
};

namespace kern {

enum class Backend { Scalar, Avx2 };
bool avx2_compiled() noexcept;          // this build has the AVX2 + FMA + F16C kernels
Backend default_backend() noexcept;     // Avx2 when compiled in, else Scalar
Backend backend() noexcept;             // process-wide current backend
void set_backend(Backend b);            // throws if Avx2 is not compiled in
const char* backend_name(Backend b) noexcept;
Backend backend_from_string(const std::string& s);  // "scalar" | "avx2" | "auto"

// Dot product of one packed row (n weights, n % 32 == 0) with x.
float dot_q4_0(const std::uint8_t* row, const float* x, std::size_t n) noexcept;
float dot_q8_0(const std::uint8_t* row, const float* x, std::size_t n) noexcept;

void matvec_q4_0(float* y, const std::uint8_t* W, const float* x, std::size_t rows, std::size_t cols) noexcept;
void matvec_q8_0(float* y, const std::uint8_t* W, const float* x, std::size_t rows, std::size_t cols) noexcept;

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#define EF_HAVE_AVX2 1
float dot_q4_0_avx2(const std::uint8_t* row, const float* x, std::size_t n) noexcept;
float dot_q8_0_avx2(const std::uint8_t* row, const float* x, std::size_t n) noexcept;
float dot_bf16_avx2(const std::uint16_t* w, const float* x, std::size_t n) noexcept;
float dot_f32_avx2(const float* w, const float* x, std::size_t n) noexcept;
#endif

// Rows [begin, end) of y = W x (y indexed by row) with the current backend. Scalar BF16 / F32 are
// exactly the Stage 2 kernels and scalar Q8_0 / Q4_0 the Stage 6 ones; F16 converts per element.
void matvec_rows(float* y, const WeightView& W, const float* x, std::size_t begin, std::size_t end) noexcept;

// y = W x; rows split across `pool` when given (nullptr = calling thread only).
void matvec(float* y, const WeightView& W, const float* x, ThreadPool* pool = nullptr) noexcept;

// Several products in ONE parallel region (e.g. q/k/v from the same input, or gate/up): fewer
// thread synchronisations per token. Each job's rows are computed exactly as by matvec().
struct MatvecJob {
  float* y;
  const WeightView* W;
  const float* x;
};
void matvec_jobs(const MatvecJob* jobs, std::size_t n_jobs, ThreadPool* pool) noexcept;

// Row r of W as FP32 (embedding lookup): exact for F32 / F16 / BF16, dequantized for Q8_0 / Q4_0.
void row_f32(float* out, const WeightView& W, std::size_t r) noexcept;

}  // namespace kern
}  // namespace ef
