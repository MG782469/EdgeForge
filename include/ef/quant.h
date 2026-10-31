// EdgeForge - Stage 5: block quantization formats.
//
// Q4_0: 32 weights -> 18 bytes (4.5 bits/weight): one FP16 scale d + 16 bytes of 4-bit codes.
//       w[j]    = ((qs[j] & 0x0F) - 8) * d      j = 0..15
//       w[j+16] = ((qs[j] >> 4)   - 8) * d      (low/high nibble split: unpack = AND + SHIFT)
// Q8_0: 32 weights -> 34 bytes (8.5 bits/weight): FP16 scale d + 32 int8 codes, w[j] = qs[j] * d.
//
// The Q4 scale is chosen per block by trying several candidates (the plain llama.cpp choice
// d = max/-8 is one of them, so the search is never worse), then refined by least squares
// (codes fixed -> d = sum(x*c)/sum(c^2), re-quantize, keep if better). Scales are rounded to
// FP16 BEFORE the codes are computed, so the stored (d, codes) pair is exactly what
// dequantization sees. With clipping and FP16 scales the error per weight stays <= ~|d|.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace ef {

inline constexpr std::size_t kQK = 32;  // weights per block

struct BlockQ4_0 {
  std::uint16_t d;       // FP16 scale
  std::uint8_t qs[16];   // 32 x 4-bit codes
};
struct BlockQ8_0 {
  std::uint16_t d;       // FP16 scale
  std::int8_t qs[32];
};
static_assert(sizeof(BlockQ4_0) == 18, "BlockQ4_0 must be packed to 18 bytes");
static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 must be packed to 34 bytes");

enum class QType : std::uint8_t { F32, F16, BF16, Q8_0, Q4_0 };

const char* qtype_name(QType t) noexcept;
std::optional<QType> qtype_from_string(std::string_view s) noexcept;
bool qtype_is_block(QType t) noexcept;
double qtype_bits_per_weight(QType t) noexcept;
// Bytes of one row of `cols` weights. Block types require cols % 32 == 0 (throws otherwise).
std::uint64_t qtype_row_bytes(QType t, std::uint64_t cols);

// ---- single block (32 weights) ---------------------------------------------
void quantize_block_q4_0(const float* x, BlockQ4_0& out, bool search = true) noexcept;
void quantize_block_q8_0(const float* x, BlockQ8_0& out) noexcept;
void dequantize_block_q4_0(const BlockQ4_0& b, float* out) noexcept;
void dequantize_block_q8_0(const BlockQ8_0& b, float* out) noexcept;

// ---- rows ------------------------------------------------------------------
// n weights -> qtype_row_bytes(t, n) bytes at `out` (F32/F16/BF16 are plain conversions).
void quantize_row(QType t, const float* x, std::size_t n, void* out, bool search = true);
void dequantize_row(QType t, const void* in, std::size_t n, float* out);

// ---- analysis --------------------------------------------------------------
// Symmetric 4-bit quantize -> dequantize of one group of any size (FP16 scale), returning
// the reconstruction. Used to compare group sizes 32/64/128 without a file format.
void fake_quant_q4_group(const float* x, std::size_t n, float* out, bool search) noexcept;

}  // namespace ef
