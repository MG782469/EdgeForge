// EdgeForge — exhaustive BF16 / FP16 conversion tests.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>

#include "ef/fp16.h"
#include "test_framework.h"

using namespace ef;

namespace {
bool is_nan_bf16(std::uint16_t h) { return (h & 0x7F80) == 0x7F80 && (h & 0x007F) != 0; }
bool is_nan_fp16(std::uint16_t h) { return (h & 0x7C00) == 0x7C00 && (h & 0x03FF) != 0; }
}  // namespace

TEST(fp16_bf16_known_values) {
  CHECK_EQ(bf16_to_f32(0x3F80), 1.0f);
  CHECK_EQ(bf16_to_f32(0xC000), -2.0f);
  CHECK_EQ(bf16_to_f32(0x3F00), 0.5f);
  CHECK_EQ(bf16_to_f32(0x4049), 3.140625f);
  CHECK_EQ(f32_to_bits(bf16_to_f32(0x0000)), 0x00000000u);  // +0
  CHECK_EQ(f32_to_bits(bf16_to_f32(0x8000)), 0x80000000u);  // -0 keeps its sign
  CHECK(std::isinf(bf16_to_f32(0x7F80)) && bf16_to_f32(0x7F80) > 0);
  CHECK(std::isinf(bf16_to_f32(0xFF80)) && bf16_to_f32(0xFF80) < 0);
  CHECK(std::isnan(bf16_to_f32(0x7FC0)));
  CHECK_EQ(bf16_to_f32(0x0001), std::ldexp(1.0f, -133));  // smallest subnormal
  CHECK_EQ(bf16_to_f32(0x7F7F), 3.38953139e38f);           // largest finite
  CHECK_EQ(f32_to_bf16(1.0f), std::uint16_t{0x3F80});
  CHECK_EQ(f32_to_bf16(-2.0f), std::uint16_t{0xC000});
}

TEST(fp16_bf16_roundtrip_all_65536) {
  int mismatches = 0;
  for (std::uint32_t h = 0; h < 65536; ++h) {
    const auto x = static_cast<std::uint16_t>(h);
    const float f = bf16_to_f32(x);
    if (f32_to_bits(f) != (h << 16)) ++mismatches;  // decode is a pure shift
    const std::uint16_t back = f32_to_bf16(f);
    if (is_nan_bf16(x)) {
      if (!is_nan_bf16(back)) ++mismatches;
    } else if (back != x) {
      ++mismatches;
    }
  }
  CHECK_EQ(mismatches, 0);
}

TEST(fp16_bf16_rounding) {
  // Ties go to even: 0x3F808000 is exactly halfway between 0x3F80 and 0x3F81.
  CHECK_EQ(f32_to_bf16(bits_to_f32(0x3F808000u)), std::uint16_t{0x3F80});
  CHECK_EQ(f32_to_bf16(bits_to_f32(0x3F818000u)), std::uint16_t{0x3F82});
  CHECK_EQ(f32_to_bf16(bits_to_f32(0x3F808001u)), std::uint16_t{0x3F81});  // just above half
  CHECK_EQ(f32_to_bf16(bits_to_f32(0x3F807FFFu)), std::uint16_t{0x3F80});  // just below half
  CHECK_EQ(f32_to_bf16(bits_to_f32(0x7F7FFFFFu)), std::uint16_t{0x7F80});  // FLT_MAX rounds to inf
  CHECK(is_nan_bf16(f32_to_bf16(bits_to_f32(0x7F800001u))));             // signalling NaN stays NaN
  CHECK(is_nan_bf16(f32_to_bf16(std::numeric_limits<float>::quiet_NaN())));
}

TEST(fp16_half_known_values) {
  CHECK_EQ(fp16_to_f32(0x3C00), 1.0f);
  CHECK_EQ(fp16_to_f32(0xC000), -2.0f);
  CHECK_EQ(fp16_to_f32(0x7BFF), 65504.0f);
  CHECK_EQ(fp16_to_f32(0x0001), std::ldexp(1.0f, -24));  // smallest subnormal
  CHECK_EQ(fp16_to_f32(0x03FF), std::ldexp(1023.0f, -24));  // largest subnormal
  CHECK_EQ(fp16_to_f32(0x0400), std::ldexp(1.0f, -14));  // smallest normal
  CHECK_EQ(f32_to_bits(fp16_to_f32(0x8000)), 0x80000000u);
  CHECK(std::isinf(fp16_to_f32(0x7C00)) && fp16_to_f32(0x7C00) > 0);
  CHECK(std::isinf(fp16_to_f32(0xFC00)) && fp16_to_f32(0xFC00) < 0);
  CHECK(std::isnan(fp16_to_f32(0x7E00)));
}

TEST(fp16_half_roundtrip_all_65536) {
  int mismatches = 0;
  for (std::uint32_t h = 0; h < 65536; ++h) {
    const auto x = static_cast<std::uint16_t>(h);
    const std::uint16_t back = f32_to_fp16(fp16_to_f32(x));
    if (is_nan_fp16(x)) {
      if (!is_nan_fp16(back)) ++mismatches;
    } else if (back != x) {
      ++mismatches;
    }
  }
  CHECK_EQ(mismatches, 0);
}

TEST(fp16_half_rounding) {
  CHECK_EQ(f32_to_fp16(65504.0f), std::uint16_t{0x7BFF});
  CHECK_EQ(f32_to_fp16(65519.0f), std::uint16_t{0x7BFF});  // below the midpoint: stays finite
  CHECK_EQ(f32_to_fp16(65520.0f), std::uint16_t{0x7C00});  // midpoint, ties-to-even -> inf
  CHECK_EQ(f32_to_fp16(1e10f), std::uint16_t{0x7C00});
  CHECK_EQ(f32_to_fp16(-1e10f), std::uint16_t{0xFC00});
  CHECK_EQ(f32_to_fp16(1.0f + std::ldexp(1.0f, -11)), std::uint16_t{0x3C00});         // tie -> even (down)
  CHECK_EQ(f32_to_fp16(1.0f + 3.0f * std::ldexp(1.0f, -11)), std::uint16_t{0x3C02});  // tie -> even (up)
  CHECK_EQ(f32_to_fp16(std::ldexp(1.0f, -25)), std::uint16_t{0x0000});  // exactly half of min subnormal -> 0
  CHECK_EQ(f32_to_fp16(std::ldexp(1.5f, -25)), std::uint16_t{0x0001});  // above half -> min subnormal
  CHECK_EQ(f32_to_fp16(std::ldexp(3.0f, -25)), std::uint16_t{0x0002});  // 1.5 ulp tie -> even (2)
  CHECK_EQ(f32_to_fp16(-std::ldexp(1.0f, -30)), std::uint16_t{0x8000});  // underflow keeps sign
  CHECK(is_nan_fp16(f32_to_fp16(std::numeric_limits<float>::quiet_NaN())));
}

TEST(fp16_matches_compiler_conversions) {
  // Cross-check against the compiler's own IEEE conversions where available.
  std::mt19937 rng(7);
  int checked = 0, mismatches = 0;
  for (int i = 0; i < 2'000'000; ++i) {
    std::uint32_t u = rng();
    if (i % 4 == 0) u = (u & 0x807FFFFFu) | ((100u + rng() % 60u) << 23);  // focus on fp16 range
    const float f = bits_to_f32(u);
    if (std::isnan(f)) continue;
#if defined(__FLT16_MANT_DIG__) && (defined(__x86_64__) || defined(__aarch64__))
    {
      const _Float16 h = static_cast<_Float16>(f);
      std::uint16_t hb;
      std::memcpy(&hb, &h, 2);
      ++checked;
      if (hb != f32_to_fp16(f)) ++mismatches;
    }
#endif
#if defined(__BFLT16_MANT_DIG__) && defined(__x86_64__) && !defined(__clang__)
    {
      const __bf16 b = static_cast<__bf16>(f);
      std::uint16_t bb;
      std::memcpy(&bb, &b, 2);
      ++checked;
      if (bb != f32_to_bf16(f)) ++mismatches;
    }
#endif
  }
  CHECK_EQ(mismatches, 0);
  if (checked == 0) std::printf("    (compiler has no _Float16/__bf16; exhaustive tests above still apply)\n");
}
