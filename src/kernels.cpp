// EdgeForge - Stage 2 scalar FP32 kernels.
#include "ef/kernels.h"

#include <cmath>
#include <cstring>
#include <limits>

#include "ef/fp16.h"

namespace ef::kern {

namespace {

// Loads one weight element as f32. Same arithmetic for both weight formats, so
// matvec_bf16 == matvec_f32(upcast(W)) exactly.
struct LoadF32 {
  const float* p;
  float operator()(std::size_t i) const noexcept { return p[i]; }
};
struct LoadBF16 {
  const std::uint16_t* p;
  float operator()(std::size_t i) const noexcept { return bf16_to_f32(p[i]); }
};

// Dot product of a weight row with x using 8 independent FP32 partial sums
// (also lets the compiler vectorise without -ffast-math), combined pairwise.
template <class Load>
inline float dot_row(Load w, const float* x, std::size_t n) noexcept {
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    a0 += w(i + 0) * x[i + 0];
    a1 += w(i + 1) * x[i + 1];
    a2 += w(i + 2) * x[i + 2];
    a3 += w(i + 3) * x[i + 3];
    a4 += w(i + 4) * x[i + 4];
    a5 += w(i + 5) * x[i + 5];
    a6 += w(i + 6) * x[i + 6];
    a7 += w(i + 7) * x[i + 7];
  }
  for (; i < n; ++i) a0 += w(i) * x[i];
  return ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
}

}  // namespace

float dot_f32(const float* a, const float* b, std::size_t n) noexcept { return dot_row(LoadF32{a}, b, n); }

void matvec_f32(float* y, const float* W, const float* x, std::size_t rows, std::size_t cols) noexcept {
  for (std::size_t r = 0; r < rows; ++r) y[r] = dot_row(LoadF32{W + r * cols}, x, cols);
}

void matvec_bf16(float* y, const std::uint16_t* W, const float* x, std::size_t rows, std::size_t cols) noexcept {
  for (std::size_t r = 0; r < rows; ++r) y[r] = dot_row(LoadBF16{W + r * cols}, x, cols);
}

void matvec_f64_ref(double* y, const float* W, const float* x, std::size_t rows, std::size_t cols) noexcept {
  for (std::size_t r = 0; r < rows; ++r) {
    double s = 0.0;
    for (std::size_t c = 0; c < cols; ++c) s += static_cast<double>(W[r * cols + c]) * static_cast<double>(x[c]);
    y[r] = s;
  }
}

void add_inplace(float* x, const float* y, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) x[i] += y[i];
}

void rmsnorm(float* out, const float* x, const float* weight, std::size_t n, float eps) noexcept {
  double ss = 0.0;
  for (std::size_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * static_cast<double>(x[i]);
  const double mean = n ? ss / static_cast<double>(n) : 0.0;
  const auto inv = static_cast<float>(1.0 / std::sqrt(mean + static_cast<double>(eps)));
  for (std::size_t i = 0; i < n; ++i) out[i] = weight[i] * (x[i] * inv);  // HF order: w * (x * rsqrt)
}

void softmax_inplace(float* x, std::size_t n) noexcept {
  if (n == 0) return;
  float mx = -std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < n; ++i) mx = x[i] > mx ? x[i] : mx;
  if (mx == -std::numeric_limits<float>::infinity()) {  // everything masked
    for (std::size_t i = 0; i < n; ++i) x[i] = 0.0f;
    return;
  }
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - mx);  // exp(-inf) == 0 for masked entries
    sum += static_cast<double>(x[i]);
  }
  const auto inv = static_cast<float>(1.0 / sum);
  for (std::size_t i = 0; i < n; ++i) x[i] *= inv;
}

float silu(float x) noexcept { return x / (1.0f + std::exp(-x)); }

void swiglu(float* out, const float* gate, const float* up, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) out[i] = silu(gate[i]) * up[i];
}

void embedding_bf16(float* out, const std::uint16_t* table, std::size_t token, std::size_t dim) noexcept {
  const std::uint16_t* row = table + token * dim;
  for (std::size_t i = 0; i < dim; ++i) out[i] = bf16_to_f32(row[i]);
}

std::size_t argmax(const float* x, std::size_t n) noexcept {
  std::size_t best = 0;
  for (std::size_t i = 1; i < n; ++i) {
    if (x[i] > x[best]) best = i;
  }
  return best;
}

RopeTable::RopeTable(std::size_t max_pos, std::size_t head_dim, float theta)
    : max_pos_(max_pos), head_dim_(head_dim) {
  const std::size_t half = head_dim / 2;
  inv_freq_.resize(half);
  for (std::size_t i = 0; i < half; ++i) {
    // torch: 1.0 / (base ** (arange(0, dim, 2).float() / dim)), all in float32
    const float expo = static_cast<float>(2 * i) / static_cast<float>(head_dim);
    inv_freq_[i] = 1.0f / std::pow(theta, expo);
  }
  cos_.resize(max_pos * half);
  sin_.resize(max_pos * half);
  for (std::size_t p = 0; p < max_pos; ++p) {
    for (std::size_t i = 0; i < half; ++i) {
      const float angle = static_cast<float>(p) * inv_freq_[i];  // float32 product, as in HF
      cos_[p * half + i] = std::cos(angle);
      sin_[p * half + i] = std::sin(angle);
    }
  }
}

void rope_apply(float* x, std::size_t n_heads, std::size_t head_dim, const float* cos_row,
                const float* sin_row) noexcept {
  const std::size_t half = head_dim / 2;
  for (std::size_t h = 0; h < n_heads; ++h) {
    float* v = x + h * head_dim;
    for (std::size_t i = 0; i < half; ++i) {
      const float a = v[i];
      const float b = v[i + half];
      // HF: x*cos + rotate_half(x)*sin, rotate_half(x) = [-x2, x1]
      v[i] = a * cos_row[i] - b * sin_row[i];
      v[i + half] = b * cos_row[i] + a * sin_row[i];
    }
  }
}

void attention(float* out, const float* q, const float* k_cache, const float* v_cache, std::size_t n_pos,
               std::size_t n_heads, std::size_t n_kv_heads, std::size_t head_dim, float* scores) noexcept {
  const std::size_t group = n_heads / n_kv_heads;
  const std::size_t kv_dim = n_kv_heads * head_dim;
  const auto scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  for (std::size_t h = 0; h < n_heads; ++h) {
    const std::size_t kv = h / group;
    const float* qh = q + h * head_dim;
    for (std::size_t t = 0; t < n_pos; ++t) {
      scores[t] = dot_f32(qh, k_cache + t * kv_dim + kv * head_dim, head_dim) * scale;
    }
    softmax_inplace(scores, n_pos);
    float* oh = out + h * head_dim;
    std::memset(oh, 0, head_dim * sizeof(float));
    for (std::size_t t = 0; t < n_pos; ++t) {
      const float p = scores[t];
      const float* vt = v_cache + t * kv_dim + kv * head_dim;
      for (std::size_t d = 0; d < head_dim; ++d) oh[d] += p * vt[d];
    }
  }
}

}  // namespace ef::kern
