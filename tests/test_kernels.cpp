// EdgeForge - Stage 2 kernel unit tests (known values, float64 references, properties).
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "ef/fp16.h"
#include "ef/kernels.h"
#include "test_framework.h"

using namespace ef;
using namespace ef::kern;

namespace {

std::vector<float> randn(std::size_t n, std::mt19937& rng, float scale = 1.0f) {
  std::normal_distribution<float> nd(0.0f, scale);
  std::vector<float> v(n);
  for (auto& x : v) x = nd(rng);
  return v;
}

template <class A, class B>
bool close(A a_in, B b_in, double rel, double abs_tol = 0.0) {
  const auto a = static_cast<double>(a_in);
  const auto b = static_cast<double>(b_in);
  return std::fabs(a - b) <= abs_tol + rel * std::max(std::fabs(a), std::fabs(b));
}

// Naive causal GQA attention in double, for one query position.
std::vector<double> attention_ref(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
                                  std::size_t n_pos, std::size_t H, std::size_t KV, std::size_t hd) {
  std::vector<double> out(H * hd, 0.0);
  for (std::size_t h = 0; h < H; ++h) {
    const std::size_t kv = h / (H / KV);
    std::vector<double> s(n_pos);
    double mx = -1e300;
    for (std::size_t t = 0; t < n_pos; ++t) {
      double d = 0;
      for (std::size_t i = 0; i < hd; ++i) d += double(q[h * hd + i]) * double(k[t * KV * hd + kv * hd + i]);
      s[t] = d / std::sqrt(double(hd));
      mx = std::max(mx, s[t]);
    }
    double sum = 0;
    for (auto& x : s) sum += (x = std::exp(x - mx));
    for (std::size_t t = 0; t < n_pos; ++t) {
      for (std::size_t i = 0; i < hd; ++i) out[h * hd + i] += s[t] / sum * double(v[t * KV * hd + kv * hd + i]);
    }
  }
  return out;
}

}  // namespace

TEST(kernels_matvec_error_within_fp32_bound) {
  std::mt19937 rng(1);
  const std::size_t shapes[][2] = {{1, 1}, {3, 7}, {5, 8}, {17, 33}, {64, 2048}, {8, 5632}, {2, 100003}};
  for (const auto& sh : shapes) {
    const std::size_t rows = sh[0], cols = sh[1];
    const auto W = randn(rows * cols, rng, 0.02f);
    const auto x = randn(cols, rng);
    std::vector<float> y(rows);
    std::vector<double> ref(rows);
    matvec_f32(y.data(), W.data(), x.data(), rows, cols);
    matvec_f64_ref(ref.data(), W.data(), x.data(), rows, cols);
    // Rigorous bound: depth of the summation tree (cols/8 + 4) * unit roundoff * sum|w*x|.
    const double depth = static_cast<double>(cols / 8 + 4);
    int bad = 0;
    for (std::size_t r = 0; r < rows; ++r) {
      double abs_sum = 0;
      for (std::size_t c = 0; c < cols; ++c) abs_sum += std::fabs(double(W[r * cols + c]) * double(x[c]));
      if (std::fabs(double(y[r]) - ref[r]) > depth * std::ldexp(1.0, -24) * abs_sum * 1.01) ++bad;
    }
    CHECK_EQ(bad, 0);
  }
  CHECK_EQ(dot_f32(std::vector<float>{1, 2, 3}.data(), std::vector<float>{4, 5, 6}.data(), 3), 32.0f);
}

TEST(kernels_matvec_bf16_equals_f32_on_upcast_weights) {
  std::mt19937 rng(2);
  const std::size_t rows = 37, cols = 2051;
  const auto Wf = randn(rows * cols, rng, 0.05f);
  std::vector<std::uint16_t> Wb(Wf.size());
  std::vector<float> Wup(Wf.size());
  for (std::size_t i = 0; i < Wf.size(); ++i) {
    Wb[i] = f32_to_bf16(Wf[i]);
    Wup[i] = bf16_to_f32(Wb[i]);
  }
  const auto x = randn(cols, rng);
  std::vector<float> y1(rows), y2(rows);
  matvec_bf16(y1.data(), Wb.data(), x.data(), rows, cols);
  matvec_f32(y2.data(), Wup.data(), x.data(), rows, cols);
  int diff = 0;
  for (std::size_t r = 0; r < rows; ++r) diff += (f32_to_bits(y1[r]) != f32_to_bits(y2[r]));
  CHECK_EQ(diff, 0);  // BF16 widening is exact, so results must be bit-identical
}

TEST(kernels_rmsnorm) {
  const std::vector<float> x = {1, 2, 3, 4}, ones = {1, 1, 1, 1}, w = {2, 0.5f, -1, 0};
  std::vector<float> o(4);
  rmsnorm(o.data(), x.data(), ones.data(), 4, 0.0f);
  const double rms = std::sqrt(7.5);
  for (int i = 0; i < 4; ++i) CHECK(close(o[i], (i + 1) / rms, 1e-6));
  rmsnorm(o.data(), x.data(), w.data(), 4, 0.0f);
  CHECK(close(o[0], 2 / rms, 1e-6) && close(o[1], 1 / rms, 1e-6) && close(o[2], -3 / rms, 1e-6) && o[3] == 0.0f);
  // eps keeps an all-zero input finite
  const std::vector<float> z = {0, 0, 0, 0};
  rmsnorm(o.data(), z.data(), ones.data(), 4, 1e-5f);
  CHECK(o[0] == 0.0f && std::isfinite(o[3]));
  // eps enters as rsqrt(mean + eps)
  const std::vector<float> small = {1e-3f, 1e-3f};
  rmsnorm(o.data(), small.data(), ones.data(), 2, 1e-6f);
  CHECK(close(o[0], 1e-3 / std::sqrt(1e-6 + 1e-6), 1e-6));
  // scale invariance (eps = 0) and in-place aliasing
  std::mt19937 rng(3);
  auto v = randn(2048, rng);
  std::vector<float> a(2048), b(2048);
  rmsnorm(a.data(), v.data(), std::vector<float>(2048, 1.0f).data(), 2048, 0.0f);
  for (auto& t : v) t *= 37.0f;
  rmsnorm(v.data(), v.data(), std::vector<float>(2048, 1.0f).data(), 2048, 0.0f);
  int bad = 0;
  for (std::size_t i = 0; i < 2048; ++i) bad += !close(a[i], v[i], 1e-5, 1e-7);
  CHECK_EQ(bad, 0);
}

TEST(kernels_softmax_stable_and_masked) {
  std::vector<float> x = {1, 2, 3};
  softmax_inplace(x.data(), 3);
  CHECK(close(x[0], 0.09003057317, 1e-6) && close(x[1], 0.24472847105, 1e-6) && close(x[2], 0.66524095577, 1e-6));
  std::vector<float> big = {1000, 1001, 1002};  // naive exp() would overflow
  softmax_inplace(big.data(), 3);
  CHECK(close(big[2], 0.66524095577, 1e-6));
  std::vector<float> neg = {-1000, -1001, -1002};  // naive exp() would underflow to 0/0
  softmax_inplace(neg.data(), 3);
  CHECK(close(neg[0], 0.66524095577, 1e-6));
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<float> m = {-inf, 0.5f, -inf};
  softmax_inplace(m.data(), 3);
  CHECK(m[0] == 0.0f && m[1] == 1.0f && m[2] == 0.0f);
  std::vector<float> all = {-inf, -inf};
  softmax_inplace(all.data(), 2);
  CHECK(all[0] == 0.0f && all[1] == 0.0f);
  std::vector<float> one = {42.0f};
  softmax_inplace(one.data(), 1);
  CHECK(one[0] == 1.0f);
  std::mt19937 rng(4);
  auto r = randn(1000, rng, 10.0f);
  softmax_inplace(r.data(), r.size());
  double s = 0;
  for (float t : r) s += double(t);
  CHECK(close(s, 1.0, 1e-5));
}

TEST(kernels_silu_swiglu) {
  CHECK(silu(0.0f) == 0.0f);
  CHECK(close(silu(1.0f), 0.7310585786, 1e-6));
  CHECK(close(silu(-1.0f), -0.2689414214, 1e-6));
  CHECK(close(silu(20.0f), 20.0, 1e-7));
  CHECK(silu(100.0f) == 100.0f);
  const float neg = silu(-100.0f);  // exp(100) overflows to inf: must give -0, not NaN
  CHECK(!std::isnan(neg) && std::fabs(neg) < 1e-30f);
  const std::vector<float> g = {1, -1, 0, 2}, u = {2, 3, 5, -1};
  std::vector<float> o(4);
  swiglu(o.data(), g.data(), u.data(), 4);
  CHECK(close(o[0], 2 * 0.7310585786, 1e-6) && close(o[1], 3 * -0.2689414214, 1e-6) && o[2] == 0.0f);
  CHECK(close(o[3], -2 / (1 + std::exp(-2.0)), 1e-6));
}

TEST(kernels_rope_matches_hf_half_split_formula) {
  const std::size_t hd = 64, H = 3;
  const float theta = 10000.0f;
  const RopeTable rt(32, hd, theta);
  CHECK_EQ(rt.inv_freq()[0], 1.0f);
  for (std::size_t i = 0; i < hd / 2; ++i) CHECK(close(rt.inv_freq()[i], std::pow(10000.0, -2.0 * i / hd), 2e-6));
  std::mt19937 rng(5);
  const auto x0 = randn(H * hd, rng);
  auto x = x0;
  rope_apply(x.data(), H, hd, rt.cos_row(0), rt.sin_row(0));
  CHECK(x == x0);  // position 0 is the identity, exactly
  const std::size_t pos = 7;
  x = x0;
  rope_apply(x.data(), H, hd, rt.cos_row(pos), rt.sin_row(pos));
  int bad = 0;
  for (std::size_t h = 0; h < H; ++h) {
    for (std::size_t i = 0; i < hd / 2; ++i) {
      const double ang = double(pos) * std::pow(10000.0, -2.0 * i / hd);
      const double a = double(x0[h * hd + i]), b = double(x0[h * hd + i + hd / 2]);  // pairs (i, i + hd/2), NOT (2i, 2i+1)
      bad += !close(x[h * hd + i], a * std::cos(ang) - b * std::sin(ang), 1e-5, 1e-6);
      bad += !close(x[h * hd + i + hd / 2], b * std::cos(ang) + a * std::sin(ang), 1e-5, 1e-6);
      // rotation preserves the length of every pair
      const double n0 = a * a + b * b;
      const double c0 = double(x[h * hd + i]), c1 = double(x[h * hd + i + hd / 2]);
      const double n1 = c0 * c0 + c1 * c1;
      bad += !close(n0, n1, 1e-5, 1e-7);
    }
  }
  CHECK_EQ(bad, 0);
}

TEST(kernels_rope_scores_depend_only_on_relative_position) {
  const std::size_t hd = 64;
  const RopeTable rt(64, hd, 10000.0f);
  std::mt19937 rng(6);
  const auto q0 = randn(hd, rng), k0 = randn(hd, rng);
  auto score = [&](std::size_t m, std::size_t n) {
    auto q = q0, k = k0;
    rope_apply(q.data(), 1, hd, rt.cos_row(m), rt.sin_row(m));
    rope_apply(k.data(), 1, hd, rt.cos_row(n), rt.sin_row(n));
    return double(dot_f32(q.data(), k.data(), hd));
  };
  CHECK(close(score(5, 3), score(40, 38), 1e-4, 1e-4));
  CHECK(close(score(9, 9), score(0, 0), 1e-4, 1e-4));
  CHECK(!close(score(5, 3), score(5, 30), 1e-3));
}

TEST(kernels_attention_causal_gqa) {
  const std::size_t H = 4, KV = 2, hd = 8, T = 5;
  std::mt19937 rng(7);
  const auto q = randn(H * hd, rng), k = randn(T * KV * hd, rng), v = randn(T * KV * hd, rng);
  std::vector<float> out(H * hd), scores(T);
  for (std::size_t n = 1; n <= T; ++n) {
    attention(out.data(), q.data(), k.data(), v.data(), n, H, KV, hd, scores.data());
    const auto ref = attention_ref(q, k, v, n, H, KV, hd);
    int bad = 0;
    for (std::size_t i = 0; i < out.size(); ++i) bad += !close(out[i], ref[i], 1e-5, 1e-6);
    CHECK_EQ(bad, 0);
  }
  // First position sees only itself: output == its own V row, exactly.
  attention(out.data(), q.data(), k.data(), v.data(), 1, H, KV, hd, scores.data());
  for (std::size_t h = 0; h < H; ++h) {
    for (std::size_t i = 0; i < hd; ++i) CHECK_EQ(out[h * hd + i], v[(h / 2) * hd + i]);
  }
  // GQA mapping: heads 0,1 read KV head 0; heads 2,3 read KV head 1.
  std::vector<float> vc(T * KV * hd);
  for (std::size_t t = 0; t < T; ++t) {
    for (std::size_t i = 0; i < hd; ++i) {
      vc[t * KV * hd + i] = 1.0f;
      vc[t * KV * hd + hd + i] = 2.0f;
    }
  }
  attention(out.data(), q.data(), k.data(), vc.data(), T, H, KV, hd, scores.data());
  CHECK(close(out[0], 1.0, 1e-6) && close(out[hd], 1.0, 1e-6) && close(out[2 * hd], 2.0, 1e-6) &&
        close(out[3 * hd], 2.0, 1e-6));
}

TEST(kernels_embedding_and_argmax) {
  const std::vector<std::uint16_t> table = {0x3F80, 0x4000, 0xBF80, 0x0000, 0x4049, 0xC0A0};  // [3][2]
  std::vector<float> o(2);
  embedding_bf16(o.data(), table.data(), 2, 2);
  CHECK(o[0] == 3.140625f && o[1] == -5.0f);
  const std::vector<float> l = {0.5f, 3.0f, -1.0f, 3.0f};
  CHECK_EQ(argmax(l.data(), l.size()), std::size_t{1});  // first max wins on ties
  CHECK_EQ(argmax(l.data(), 1), std::size_t{0});
}
