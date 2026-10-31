// EdgeForge — exact 16-bit float conversions (header-only, no intrinsics).
//
//  * bf16 -> f32 is exact (a 16-bit shift).
//  * fp16 -> f32 is exact (handles subnormals, inf, NaN).
//  * f32 -> bf16 / fp16 round to nearest, ties to even, and keep NaN a NaN.
// Verified exhaustively over all 65,536 bit patterns in tests/test_fp16.cpp.
#pragma once

#include <cstdint>
#include <cstring>

namespace ef {

inline std::uint32_t f32_to_bits(float f) noexcept {
  std::uint32_t u;
  std::memcpy(&u, &f, sizeof u);
  return u;
}

inline float bits_to_f32(std::uint32_t u) noexcept {
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

// ---- bfloat16 -------------------------------------------------------------

inline float bf16_to_f32(std::uint16_t h) noexcept {
  return bits_to_f32(static_cast<std::uint32_t>(h) << 16);
}

inline std::uint16_t f32_to_bf16(float f) noexcept {
  std::uint32_t u = f32_to_bits(f);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) {  // NaN: truncate, force quiet bit
    return static_cast<std::uint16_t>((u >> 16) | 0x0040u);
  }
  const std::uint32_t lsb = (u >> 16) & 1u;
  u += 0x7FFFu + lsb;  // round to nearest even (overflow correctly becomes inf)
  return static_cast<std::uint16_t>(u >> 16);
}

// ---- IEEE binary16 ---------------------------------------------------------

inline float fp16_to_f32(std::uint16_t h) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1Fu;
  std::uint32_t mant = h & 0x3FFu;
  std::uint32_t u;
  if (exp == 0) {
    if (mant == 0) {
      u = sign;  // +-0
    } else {     // subnormal: normalise
      std::uint32_t e = 0;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        ++e;
      }
      mant &= 0x3FFu;
      u = sign | ((113u - e) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    u = sign | 0x7F800000u | (mant << 13);  // inf / NaN (payload kept)
  } else {
    u = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  return bits_to_f32(u);
}

inline std::uint16_t f32_to_fp16(float f) noexcept {
  const std::uint32_t u = f32_to_bits(f);
  const auto sign = static_cast<std::uint16_t>((u >> 16) & 0x8000u);
  const std::uint32_t a = u & 0x7FFFFFFFu;
  if (a > 0x7F800000u) {  // NaN -> quiet NaN, keep top payload bits
    return static_cast<std::uint16_t>(sign | 0x7E00u | ((a >> 13) & 0x3FFu));
  }
  if (a >= 0x477FF000u) return static_cast<std::uint16_t>(sign | 0x7C00u);  // >= 65520 -> inf
  if (a < 0x38800000u) {                                                    // below 2^-14
    if (a <= 0x33000000u) return sign;  // <= 2^-25 rounds (ties-to-even) to 0
    const std::uint32_t e = a >> 23;    // 102..112
    const std::uint32_t m = (a & 0x7FFFFFu) | 0x800000u;
    const std::uint32_t shift = 126u - e;  // 14..24
    std::uint32_t r = m >> shift;
    const std::uint32_t rem = m & ((1u << shift) - 1u);
    const std::uint32_t half = 1u << (shift - 1u);
    if (rem > half || (rem == half && (r & 1u))) ++r;
    return static_cast<std::uint16_t>(sign | r);
  }
  const std::uint32_t x = a - 0x38000000u;  // rebias exponent 127 -> 15
  const std::uint32_t r = (x + 0xFFFu + ((x >> 13) & 1u)) >> 13;
  return static_cast<std::uint16_t>(sign | r);
}

}  // namespace ef
