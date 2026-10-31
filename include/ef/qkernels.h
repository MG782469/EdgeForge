// EdgeForge - Stage 6: GEMV straight from the packed .efq weights.
//
// The quantized engine never builds a dequantized copy of a matrix: every dot product reads the
// 18-byte Q4_0 / 34-byte Q8_0 blocks in place (memory-mapped) and expands one block at a time
// into registers. Per block of 32 weights:
//     Q4_0:  d * sum_j (code_j - 8) * x_j        Q8_0:  d * sum_j q_j * x_j
// so the scale multiplies once per block instead of once per weight. FP32 activations and
// accumulation; scalar C++ that the compiler may auto-vectorise (explicit SIMD + threads: Stage 7).
#pragma once

#include <cstddef>
#include <cstdint>

#include "ef/quant.h"

namespace ef {

// A row-major [rows][cols] weight matrix in any storage type, viewed in place (zero-copy).
struct WeightView {
  QType type = QType::BF16;
  const std::uint8_t* data = nullptr;
  std::size_t rows = 0, cols = 0;
  std::size_t row_bytes() const { return static_cast<std::size_t>(qtype_row_bytes(type, cols)); }
  std::uint64_t bytes() const { return static_cast<std::uint64_t>(row_bytes()) * rows; }
};

namespace kern {

// Dot product of one packed row (n weights, n % 32 == 0) with x.
float dot_q4_0(const std::uint8_t* row, const float* x, std::size_t n) noexcept;
float dot_q8_0(const std::uint8_t* row, const float* x, std::size_t n) noexcept;

void matvec_q4_0(float* y, const std::uint8_t* W, const float* x, std::size_t rows, std::size_t cols) noexcept;
void matvec_q8_0(float* y, const std::uint8_t* W, const float* x, std::size_t rows, std::size_t cols) noexcept;

// y = W x for any storage type: BF16 and F32 use exactly the Stage 2 kernels (bit-identical to the
// reference engine); F16 converts per element; Q8_0 / Q4_0 use the packed kernels above.
void matvec(float* y, const WeightView& W, const float* x) noexcept;

// Row r of W as FP32 (embedding lookup): exact for F32 / F16 / BF16, dequantized for Q8_0 / Q4_0.
void row_f32(float* out, const WeightView& W, std::size_t r) noexcept;

}  // namespace kern
}  // namespace ef
