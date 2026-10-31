// EdgeForge — small string formatting helpers (printf-style, portable).
#pragma once

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <string>

namespace ef {

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 1, 2)))
#endif
inline std::string strf(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  va_list ap2;
  va_copy(ap2, ap);
  const int n = std::vsnprintf(nullptr, 0, fmt, ap);
  va_end(ap);
  std::string out(n > 0 ? static_cast<std::size_t>(n) : 0, '\0');
  if (n > 0) std::vsnprintf(out.data(), out.size() + 1, fmt, ap2);
  va_end(ap2);
  return out;
}

inline std::string with_commas(std::uint64_t v) {
  std::string s = std::to_string(v);
  for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) s.insert(static_cast<std::size_t>(i), ",");
  return s;
}

inline std::string human_bytes(std::uint64_t b) {
  const double v = static_cast<double>(b);
  if (b >= (1ull << 30)) return strf("%.2f GiB", v / static_cast<double>(1ull << 30));
  if (b >= (1ull << 20)) return strf("%.1f MiB", v / static_cast<double>(1ull << 20));
  if (b >= (1ull << 10)) return strf("%.1f KiB", v / 1024.0);
  return strf("%llu B", static_cast<unsigned long long>(b));
}

}  // namespace ef
