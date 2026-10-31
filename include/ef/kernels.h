// EdgeForge - Stage 2: scalar FP32 transformer kernels (correctness reference).
//
// Conventions (must match Hugging Face Llama exactly):
//  * Weight matrices are row-major [out_features][in_features], as stored in the
//    checkpoint, so y = W x is one dot product per output row.
//  * All arithmetic is FP32 with FP32 accumulation (8 independent partial sums
//    per dot product, combined pairwise). Weights may be FP32 or BF16; BF16 is
//    widened exactly (bits << 16), so a BF16 GEMV equals the FP32 GEMV on the
//    upcast matrix bit for bit.
//  * RoPE is the HF "rotate_half" (half-split) form: dimension i rotates with
//    dimension i + head_dim/2, angle = pos * theta^(-2i/head_dim).
//  * GQA: query head h reads KV head h / (n_heads / n_kv_heads).
//
// These are the reference kernels every faster kernel (threads, SIMD, INT4) is
// later tested against. They favour clarity over speed.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ef::kern {

// ---- linear algebra --------------------------------------------------------

float dot_f32(const float* a, const float* b, std::size_t n) noexcept;

// y[r] = sum_c W[r][c] * x[c]   for r in [0, rows)
void matvec_f32(float* y, const float* W, const float* x, std::size_t rows, std::size_t cols) noexcept;
void matvec_bf16(float* y, const std::uint16_t* W, const float* x, std::size_t rows, std::size_t cols) noexcept;

// Reference GEMV with double accumulation (tests only: measures FP32 error).
void matvec_f64_ref(double* y, const float* W, const float* x, std::size_t rows, std::size_t cols) noexcept;

// ---- elementwise -----------------------------------------------------------

// x += y   (residual connection)
void add_inplace(float* x, const float* y, std::size_t n) noexcept;

// out = weight * (x / sqrt(mean(x^2) + eps))   (LlamaRMSNorm; out may alias x)
void rmsnorm(float* out, const float* x, const float* weight, std::size_t n, float eps) noexcept;

// In place, numerically stable: subtracts the max before exp. -inf entries
// (masked positions) become exactly 0. If every entry is -inf, all become 0.
void softmax_inplace(float* x, std::size_t n) noexcept;

// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
float silu(float x) noexcept;

// out = silu(gate) * up   (SwiGLU; out may alias gate or up)
void swiglu(float* out, const float* gate, const float* up, std::size_t n) noexcept;

// out = row `token` of a BF16 embedding table [vocab][dim], widened to f32 (exact).
void embedding_bf16(float* out, const std::uint16_t* table, std::size_t token, std::size_t dim) noexcept;

// Index of the largest value (first one on ties). n must be > 0.
std::size_t argmax(const float* x, std::size_t n) noexcept;

// ---- RoPE ------------------------------------------------------------------

// Precomputed cos/sin, [max_pos][head_dim/2], computed exactly like HF in FP32:
// inv_freq[i] = 1 / theta^(2i/head_dim); angle = float(pos) * inv_freq[i].
class RopeTable {
 public:
  RopeTable() = default;
  RopeTable(std::size_t max_pos, std::size_t head_dim, float theta);

  std::size_t max_pos() const noexcept { return max_pos_; }
  std::size_t head_dim() const noexcept { return head_dim_; }
  const float* inv_freq() const noexcept { return inv_freq_.data(); }
  const float* cos_row(std::size_t pos) const noexcept { return cos_.data() + pos * (head_dim_ / 2); }
  const float* sin_row(std::size_t pos) const noexcept { return sin_.data() + pos * (head_dim_ / 2); }

 private:
  std::size_t max_pos_ = 0, head_dim_ = 0;
  std::vector<float> inv_freq_, cos_, sin_;
};

// Rotates n_heads consecutive heads of x ([n_heads][head_dim]) in place using
// the cos/sin row of one position (half-split pairing).
void rope_apply(float* x, std::size_t n_heads, std::size_t head_dim, const float* cos_row,
                const float* sin_row) noexcept;

// ---- attention -------------------------------------------------------------

// Causal multi-head attention with grouped KV heads for ONE query position.
//   q:        [n_heads * head_dim]               (already RoPE'd)
//   k_cache:  [n_pos][n_kv_heads * head_dim]     (already RoPE'd; positions 0..n_pos-1)
//   v_cache:  [n_pos][n_kv_heads * head_dim]
//   out:      [n_heads * head_dim]
//   scores:   scratch, at least n_pos floats
// The query attends to all n_pos cached positions (the caller passes n_pos = pos + 1,
// which is exactly the causal mask). Scores are scaled by 1/sqrt(head_dim).
void attention(float* out, const float* q, const float* k_cache, const float* v_cache, std::size_t n_pos,
               std::size_t n_heads, std::size_t n_kv_heads, std::size_t head_dim, float* scores) noexcept;
// Stage 7: the same computation for heads [h_begin, h_end) only (heads are independent, so a
// thread pool can split them; each head is computed exactly as by attention()).
void attention_heads(float* out, const float* q, const float* k_cache, const float* v_cache, std::size_t n_pos,
                     std::size_t n_heads, std::size_t n_kv_heads, std::size_t head_dim, float* scores,
                     std::size_t h_begin, std::size_t h_end) noexcept;

}  // namespace ef::kern
