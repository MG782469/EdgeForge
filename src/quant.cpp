// EdgeForge - Stage 5 block quantization.
#include "ef/quant.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <string>

#include "ef/error.h"
#include "ef/fp16.h"

namespace ef {

namespace {

constexpr float kFp16Max = 65504.0f;

// Rounds a scale to FP16 and back, so codes are computed against the stored value.
float fp16_round(float d) noexcept { return fp16_to_f32(f32_to_fp16(d)); }

int clamp_int(int v, int lo, int hi) noexcept { return v < lo ? lo : (v > hi ? hi : v); }

// Core 4-bit quantizer for one group of n values (n <= 256). Writes codes 0..15 and returns
// the FP16-rounded scale actually used. Codes decode as (q - 8) * d.
float q4_group(const float* x, std::size_t n, bool search, std::uint8_t* q) noexcept {
  float amax = 0.0f, signed_max = 0.0f;
  for (std::size_t j = 0; j < n; ++j) {
    const float a = std::fabs(x[j]);
    if (a > amax) {
      amax = a;
      signed_max = x[j];
    }
  }
  if (!(amax > 0.0f) || !std::isfinite(amax)) {
    for (std::size_t j = 0; j < n; ++j) q[j] = 8;
    return 0.0f;
  }
  // Candidate scales: llama.cpp's (largest-magnitude value maps exactly onto -8), plus
  // symmetric ones that clip a little more or less.
  float cand[6];
  int n_cand = 0;
  cand[n_cand++] = signed_max / -8.0f;
  if (search) {
    for (const float k : {7.0f, 7.25f, 7.5f, 7.75f, 8.0f}) cand[n_cand++] = amax / k;
  }
  float best_d = 0.0f;
  double best_err = -1.0;
  std::uint8_t tmp[256];
  for (int c = 0; c < n_cand; ++c) {
    const float d = fp16_round(std::clamp(cand[c], -kFp16Max, kFp16Max));
    if (d == 0.0f) continue;
    const float id = 1.0f / d;
    double err = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      const int v = clamp_int(static_cast<int>(std::nearbyint(x[j] * id)) + 8, 0, 15);
      tmp[j] = static_cast<std::uint8_t>(v);
      const double e = static_cast<double>(x[j]) - static_cast<double>(static_cast<float>(v - 8) * d);
      err += e * e;
    }
    if (best_err < 0.0 || err < best_err) {
      best_err = err;
      best_d = d;
      std::memcpy(q, tmp, n);
    }
  }
  if (best_err < 0.0) {  // every candidate underflowed FP16: block is effectively zero
    for (std::size_t j = 0; j < n; ++j) q[j] = 8;
    return 0.0f;
  }
  if (search) {
    // Least-squares refinement: with the codes fixed, the best scale is sum(x*c)/sum(c*c).
    // Re-quantize with it and keep the result only if the error drops (2 rounds).
    for (int iter = 0; iter < 2; ++iter) {
      double sxc = 0.0, scc = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        const double c = static_cast<double>(static_cast<int>(q[j]) - 8);
        sxc += static_cast<double>(x[j]) * c;
        scc += c * c;
      }
      if (scc == 0.0) break;
      const float d = fp16_round(static_cast<float>(std::clamp(sxc / scc, -65504.0, 65504.0)));
      if (d == 0.0f || d == best_d) break;
      const float id = 1.0f / d;
      double err = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        const int v = clamp_int(static_cast<int>(std::nearbyint(x[j] * id)) + 8, 0, 15);
        tmp[j] = static_cast<std::uint8_t>(v);
        const double e = static_cast<double>(x[j]) - static_cast<double>(static_cast<float>(v - 8) * d);
        err += e * e;
      }
      if (!(err < best_err)) break;
      best_err = err;
      best_d = d;
      std::memcpy(q, tmp, n);
    }
  }
  return best_d;
}

}  // namespace

const char* qtype_name(QType t) noexcept {
  switch (t) {
    case QType::F32: return "F32";
    case QType::F16: return "F16";
    case QType::BF16: return "BF16";
    case QType::Q8_0: return "Q8_0";
    case QType::Q4_0: return "Q4_0";
  }
  return "?";
}

std::optional<QType> qtype_from_string(std::string_view s) noexcept {
  for (QType t : {QType::F32, QType::F16, QType::BF16, QType::Q8_0, QType::Q4_0}) {
    std::string lower = qtype_name(t);
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (s == qtype_name(t) || s == lower) return t;
  }
  return std::nullopt;
}

bool qtype_is_block(QType t) noexcept { return t == QType::Q8_0 || t == QType::Q4_0; }

double qtype_bits_per_weight(QType t) noexcept {
  switch (t) {
    case QType::F32: return 32.0;
    case QType::F16:
    case QType::BF16: return 16.0;
    case QType::Q8_0: return 8.0 * sizeof(BlockQ8_0) / kQK;
    case QType::Q4_0: return 8.0 * sizeof(BlockQ4_0) / kQK;
  }
  return 0.0;
}

std::uint64_t qtype_row_bytes(QType t, std::uint64_t cols) {
  switch (t) {
    case QType::F32: return cols * 4;
    case QType::F16:
    case QType::BF16: return cols * 2;
    case QType::Q8_0:
    case QType::Q4_0:
      if (cols % kQK != 0) {
        throw Error(std::string(qtype_name(t)) + " needs a row length divisible by 32, got " + std::to_string(cols));
      }
      return cols / kQK * (t == QType::Q8_0 ? sizeof(BlockQ8_0) : sizeof(BlockQ4_0));
  }
  return 0;
}

void quantize_block_q4_0(const float* x, BlockQ4_0& out, bool search) noexcept {
  std::uint8_t q[kQK];
  const float d = q4_group(x, kQK, search, q);
  out.d = f32_to_fp16(d);
  for (std::size_t j = 0; j < 16; ++j) out.qs[j] = static_cast<std::uint8_t>(q[j] | (q[j + 16] << 4));
}

void dequantize_block_q4_0(const BlockQ4_0& b, float* out) noexcept {
  const float d = fp16_to_f32(b.d);
  for (std::size_t j = 0; j < 16; ++j) {
    out[j] = static_cast<float>((b.qs[j] & 0x0F) - 8) * d;
    out[j + 16] = static_cast<float>((b.qs[j] >> 4) - 8) * d;
  }
}

void quantize_block_q8_0(const float* x, BlockQ8_0& out) noexcept {
  float amax = 0.0f;
  for (std::size_t j = 0; j < kQK; ++j) amax = std::max(amax, std::fabs(x[j]));
  const float d = (amax > 0.0f && std::isfinite(amax)) ? fp16_round(std::min(amax / 127.0f, kFp16Max)) : 0.0f;
  out.d = f32_to_fp16(d);
  const float id = d != 0.0f ? 1.0f / d : 0.0f;
  for (std::size_t j = 0; j < kQK; ++j) {
    out.qs[j] = static_cast<std::int8_t>(clamp_int(static_cast<int>(std::nearbyint(x[j] * id)), -127, 127));
  }
}

void dequantize_block_q8_0(const BlockQ8_0& b, float* out) noexcept {
  const float d = fp16_to_f32(b.d);
  for (std::size_t j = 0; j < kQK; ++j) out[j] = static_cast<float>(b.qs[j]) * d;
}

void quantize_row(QType t, const float* x, std::size_t n, void* out, bool search) {
  auto* o = static_cast<std::uint8_t*>(out);
  switch (t) {
    case QType::F32:
      std::memcpy(o, x, n * 4);
      return;
    case QType::F16:
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint16_t h = f32_to_fp16(x[i]);
        std::memcpy(o + 2 * i, &h, 2);
      }
      return;
    case QType::BF16:
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint16_t h = f32_to_bf16(x[i]);
        std::memcpy(o + 2 * i, &h, 2);
      }
      return;
    case QType::Q8_0:
    case QType::Q4_0: {
      (void)qtype_row_bytes(t, n);  // validates n % 32
      for (std::size_t b = 0; b < n / kQK; ++b) {
        if (t == QType::Q4_0) {
          BlockQ4_0 blk;
          quantize_block_q4_0(x + b * kQK, blk, search);
          std::memcpy(o + b * sizeof(BlockQ4_0), &blk, sizeof blk);
        } else {
          BlockQ8_0 blk;
          quantize_block_q8_0(x + b * kQK, blk);
          std::memcpy(o + b * sizeof(BlockQ8_0), &blk, sizeof blk);
        }
      }
      return;
    }
  }
}

void dequantize_row(QType t, const void* in, std::size_t n, float* out) {
  const auto* p = static_cast<const std::uint8_t*>(in);
  switch (t) {
    case QType::F32:
      std::memcpy(out, p, n * 4);
      return;
    case QType::F16:
    case QType::BF16:
      for (std::size_t i = 0; i < n; ++i) {
        std::uint16_t h;
        std::memcpy(&h, p + 2 * i, 2);
        out[i] = t == QType::F16 ? fp16_to_f32(h) : bf16_to_f32(h);
      }
      return;
    case QType::Q8_0:
    case QType::Q4_0:
      (void)qtype_row_bytes(t, n);
      for (std::size_t b = 0; b < n / kQK; ++b) {
        if (t == QType::Q4_0) {
          BlockQ4_0 blk;
          std::memcpy(&blk, p + b * sizeof(BlockQ4_0), sizeof blk);
          dequantize_block_q4_0(blk, out + b * kQK);
        } else {
          BlockQ8_0 blk;
          std::memcpy(&blk, p + b * sizeof(BlockQ8_0), sizeof blk);
          dequantize_block_q8_0(blk, out + b * kQK);
        }
      }
      return;
  }
}

void fake_quant_q4_group(const float* x, std::size_t n, float* out, bool search) noexcept {
  std::uint8_t q[256];
  const std::size_t m = std::min<std::size_t>(n, 256);
  const float d = q4_group(x, m, search, q);
  for (std::size_t j = 0; j < m; ++j) out[j] = static_cast<float>(q[j] - 8) * d;
}

}  // namespace ef
