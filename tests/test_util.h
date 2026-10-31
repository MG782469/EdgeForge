// EdgeForge — helpers shared by tests: build safetensors bytes, temp files.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "ef/fp16.h"

namespace ef_test {

// [u64 LE header length][header][data]. len_override replaces the length field.
inline std::vector<std::uint8_t> make_safetensors(const std::string& header, const std::vector<std::uint8_t>& data,
                                                  long long len_override = -1) {
  std::vector<std::uint8_t> f(8);
  const auto n = static_cast<std::uint64_t>(len_override >= 0 ? len_override : static_cast<long long>(header.size()));
  for (int i = 0; i < 8; ++i) f[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(n >> (8 * i));
  f.insert(f.end(), header.begin(), header.end());
  f.insert(f.end(), data.begin(), data.end());
  return f;
}

template <class T>
void append_raw(std::vector<std::uint8_t>& out, T v) {
  std::uint8_t b[sizeof(T)];
  std::memcpy(b, &v, sizeof(T));
  out.insert(out.end(), b, b + sizeof(T));
}

class TempFile {
 public:
  explicit TempFile(const std::vector<std::uint8_t>& bytes, const std::string& suffix = ".safetensors") {
    static std::atomic<int> counter{0};
    path_ = std::filesystem::temp_directory_path() /
            ("edgeforge_test_" + std::to_string(counter++) + "_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)) + suffix);
    std::ofstream out(path_, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  }
  explicit TempFile(const std::string& text, const std::string& suffix)
      : TempFile(std::vector<std::uint8_t>(text.begin(), text.end()), suffix) {}
  ~TempFile() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

}  // namespace ef_test
