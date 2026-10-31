// EdgeForge — machine description for benchmark reports.
#pragma once

#include <cstdint>
#include <string>

namespace ef {

struct SystemInfo {
  std::string os;
  std::string cpu_model;
  int logical_cpus = 0;
  int physical_cores = -1;            // -1 = unknown
  std::uint64_t total_ram_bytes = 0;  // 0 = unknown
  std::uint64_t avail_ram_bytes = 0;  // 0 = unknown
  // CPU features usable by this process (OS support for AVX state is checked).
  bool sse42 = false, avx = false, avx2 = false, fma = false, f16c = false;
  bool avx512f = false, avx_vnni = false, neon = false;
  std::string compiler;
  std::string build_type;
  std::string build_flags;
};

SystemInfo query_system_info();
std::string cpu_feature_string(const SystemInfo& s);

// Peak resident set size of this process in bytes (0 if unknown).
std::uint64_t peak_rss_bytes();

}  // namespace ef
