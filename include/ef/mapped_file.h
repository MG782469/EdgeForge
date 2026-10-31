// EdgeForge — read-only memory-mapped file (POSIX mmap / Windows MapViewOfFile).
//
// Mapping the 2.2 GB checkpoint costs no RAM up front: the OS pages in only
// the bytes we actually touch, so inspecting one tensor reads one tensor.
#pragma once

#include <cstdint>
#include <filesystem>

namespace ef {

class MappedFile {
 public:
  MappedFile() = default;
  explicit MappedFile(const std::filesystem::path& path);  // throws ef::Error
  ~MappedFile();

  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  const std::uint8_t* data() const noexcept { return static_cast<const std::uint8_t*>(addr_); }
  std::uint64_t size() const noexcept { return size_; }
  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  void release() noexcept;

  std::filesystem::path path_;
  void* addr_ = nullptr;
  std::uint64_t size_ = 0;
#ifdef _WIN32
  void* file_handle_ = nullptr;
  void* mapping_handle_ = nullptr;
#else
  int fd_ = -1;
#endif
};

}  // namespace ef
