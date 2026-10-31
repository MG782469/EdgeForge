// EdgeForge - Stage 6: packed Q4_0 / Q8_0 GEMV kernels.
#include "ef/qkernels.h"

#include <cstring>

#include "ef/fp16.h"
#include "ef/kernels.h"

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

void matvec(float* y, const WeightView& W, const float* x) noexcept {
  switch (W.type) {
    case QType::BF16:
      matvec_bf16(y, reinterpret_cast<const std::uint16_t*>(W.data), x, W.rows, W.cols);
      return;
    case QType::F32:
      matvec_f32(y, reinterpret_cast<const float*>(W.data), x, W.rows, W.cols);
      return;
    case QType::Q4_0:
      matvec_q4_0(y, W.data, x, W.rows, W.cols);
      return;
    case QType::Q8_0:
      matvec_q8_0(y, W.data, x, W.rows, W.cols);
      return;
    case QType::F16: {
      for (std::size_t r = 0; r < W.rows; ++r) {
        const std::uint8_t* p = W.data + r * W.cols * 2;
        float acc = 0.0f;
        for (std::size_t c = 0; c < W.cols; ++c) {
          std::uint16_t h;
          std::memcpy(&h, p + 2 * c, 2);
          acc += fp16_to_f32(h) * x[c];
        }
        y[r] = acc;
      }
      return;
    }
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
