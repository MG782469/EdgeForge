// EdgeForge — machine description (CPU model, cores, RAM, features, build).
#include "ef/sysinfo.h"

#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <thread>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#include <vector>
#elif defined(__APPLE__)
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#else
#include <sys/resource.h>
#include <sys/utsname.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define EF_X86 1
#if defined(_MSC_VER) && !defined(__clang__)
#include <immintrin.h>
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#ifndef EF_BUILD_TYPE
#define EF_BUILD_TYPE "unknown"
#endif
#ifndef EF_BUILD_FLAGS
#define EF_BUILD_FLAGS "unknown"
#endif

namespace ef {

namespace {

#ifdef EF_X86
void cpuid(unsigned leaf, unsigned sub, unsigned r[4]) {
#if defined(_MSC_VER) && !defined(__clang__)
  int regs[4];
  __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(sub));
  for (int i = 0; i < 4; ++i) r[i] = static_cast<unsigned>(regs[i]);
#else
  r[0] = r[1] = r[2] = r[3] = 0;
  __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}

unsigned long long xgetbv0() {
#if defined(_MSC_VER) && !defined(__clang__)
  return _xgetbv(0);
#else
  unsigned eax = 0, edx = 0;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  return (static_cast<unsigned long long>(edx) << 32) | eax;
#endif
}

void detect_x86(SystemInfo& s) {
  unsigned r[4];
  cpuid(0, 0, r);
  const unsigned max_leaf = r[0];
  cpuid(0x80000000u, 0, r);
  if (r[0] >= 0x80000004u) {
    char brand[49] = {0};
    for (unsigned i = 0; i < 3; ++i) {
      cpuid(0x80000002u + i, 0, r);
      std::memcpy(brand + 16 * i, r, 16);
    }
    std::string b(brand);
    const auto first = b.find_first_not_of(' ');
    const auto last = b.find_last_not_of(' ');
    if (first != std::string::npos) s.cpu_model = b.substr(first, last - first + 1);
  }
  if (max_leaf < 1) return;
  cpuid(1, 0, r);
  const unsigned ecx1 = r[2];
  s.sse42 = (ecx1 >> 20) & 1u;
  const bool osxsave = (ecx1 >> 27) & 1u;
  const bool ymm_ok = osxsave && ((xgetbv0() & 0x6u) == 0x6u);
  const bool zmm_ok = osxsave && ((xgetbv0() & 0xE6u) == 0xE6u);
  s.avx = ymm_ok && ((ecx1 >> 28) & 1u);
  s.fma = ymm_ok && ((ecx1 >> 12) & 1u);
  s.f16c = ymm_ok && ((ecx1 >> 29) & 1u);
  if (max_leaf >= 7) {
    cpuid(7, 0, r);
    s.avx2 = ymm_ok && ((r[1] >> 5) & 1u);
    s.avx512f = zmm_ok && ((r[1] >> 16) & 1u);
    cpuid(7, 1, r);
    s.avx_vnni = ymm_ok && ((r[0] >> 4) & 1u);
  }
}
#endif

std::string compiler_string() {
#if defined(__clang__)
  return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
  return std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
  return "msvc " + std::to_string(_MSC_FULL_VER);
#else
  return "unknown";
#endif
}

}  // namespace

SystemInfo query_system_info() {
  SystemInfo s;
  s.logical_cpus = static_cast<int>(std::thread::hardware_concurrency());
  s.compiler = compiler_string();
  s.build_type = EF_BUILD_TYPE;
  s.build_flags = EF_BUILD_FLAGS;
#ifdef EF_X86
  detect_x86(s);
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
  s.neon = true;
#endif

#if defined(_WIN32)
  s.os = "Windows";
  {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len > 0) {
      std::vector<unsigned char> buf(len);
      auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
      if (GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) {
        int cores = 0;
        for (DWORD off = 0; off < len;) {
          auto* item = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
          if (item->Relationship == RelationProcessorCore) ++cores;
          off += item->Size;
        }
        s.physical_cores = cores;
      }
    }
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) {
      s.total_ram_bytes = ms.ullTotalPhys;
      s.avail_ram_bytes = ms.ullAvailPhys;
    }
  }
#elif defined(__APPLE__)
  s.os = "macOS";
  {
    char buf[256];
    size_t len = sizeof buf;
    if (s.cpu_model.empty() && sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0) {
      s.cpu_model = buf;
    }
    int cores = 0;
    len = sizeof cores;
    if (sysctlbyname("hw.physicalcpu", &cores, &len, nullptr, 0) == 0) s.physical_cores = cores;
    std::uint64_t mem = 0;
    len = sizeof mem;
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) s.total_ram_bytes = mem;
  }
#else
  {
    struct utsname u {};
    s.os = (uname(&u) == 0) ? std::string(u.sysname) + " " + u.release : "Linux";
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    std::set<std::pair<std::string, std::string>> cores;
    std::string phys = "0";
    while (std::getline(cpuinfo, line)) {
      const auto colon = line.find(':');
      if (colon == std::string::npos) continue;
      std::string key = line.substr(0, colon);
      while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) key.pop_back();
      std::string val = line.substr(colon + 1);
      if (!val.empty() && val.front() == ' ') val.erase(0, 1);
      if (key == "model name" && s.cpu_model.empty()) s.cpu_model = val;
      if (key == "physical id") phys = val;
      if (key == "core id") cores.emplace(phys, val);
    }
    if (!cores.empty()) s.physical_cores = static_cast<int>(cores.size());
    std::ifstream meminfo("/proc/meminfo");
    while (std::getline(meminfo, line)) {
      std::istringstream ls(line);
      std::string key;
      std::uint64_t kb = 0;
      ls >> key >> kb;
      if (key == "MemTotal:") s.total_ram_bytes = kb * 1024;
      if (key == "MemAvailable:") s.avail_ram_bytes = kb * 1024;
    }
  }
#endif
  if (s.cpu_model.empty()) s.cpu_model = "unknown";
  return s;
}

std::string cpu_feature_string(const SystemInfo& s) {
  std::string f;
  auto add = [&](bool on, const char* name) {
    if (on) f += (f.empty() ? "" : " ") + std::string(name);
  };
  add(s.sse42, "SSE4.2");
  add(s.avx, "AVX");
  add(s.avx2, "AVX2");
  add(s.fma, "FMA");
  add(s.f16c, "F16C");
  add(s.avx_vnni, "AVX-VNNI");
  add(s.avx512f, "AVX-512F");
  add(s.neon, "NEON");
  return f.empty() ? "none detected" : f;
}

std::uint64_t peak_rss_bytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) return pmc.PeakWorkingSetSize;
  return 0;
#else
  struct rusage ru {};
  if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#if defined(__APPLE__)
  return static_cast<std::uint64_t>(ru.ru_maxrss);  // bytes on macOS
#else
  return static_cast<std::uint64_t>(ru.ru_maxrss) * 1024;  // KiB on Linux
#endif
#endif
}

}  // namespace ef
