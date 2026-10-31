// EdgeForge — memory-mapped file implementation.
#include "ef/mapped_file.h"

#include <string>
#include <utility>

#include "ef/error.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ef {

#ifdef _WIN32

namespace {
std::string last_error_message() {
  const DWORD code = GetLastError();
  char buf[512] = {0};
  FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code, 0, buf,
                 sizeof buf, nullptr);
  std::string msg(buf);
  while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r' || msg.back() == ' ')) msg.pop_back();
  return msg + " (Windows error " + std::to_string(code) + ")";
}
}  // namespace

MappedFile::MappedFile(const std::filesystem::path& path) : path_(path) {
  HANDLE f = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
  if (f == INVALID_HANDLE_VALUE) {
    throw Error("cannot open '" + path.string() + "': " + last_error_message());
  }
  file_handle_ = f;
  LARGE_INTEGER sz;
  if (!GetFileSizeEx(f, &sz)) {
    const std::string msg = last_error_message();
    release();
    throw Error("cannot get size of '" + path.string() + "': " + msg);
  }
  size_ = static_cast<std::uint64_t>(sz.QuadPart);
  if (size_ == 0) return;  // nothing to map
  HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (m == nullptr) {
    const std::string msg = last_error_message();
    release();
    throw Error("cannot map '" + path.string() + "': " + msg);
  }
  mapping_handle_ = m;
  addr_ = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
  if (addr_ == nullptr) {
    const std::string msg = last_error_message();
    release();
    throw Error("cannot map view of '" + path.string() + "': " + msg);
  }
}

void MappedFile::release() noexcept {
  if (addr_ != nullptr) UnmapViewOfFile(addr_);
  if (mapping_handle_ != nullptr) CloseHandle(static_cast<HANDLE>(mapping_handle_));
  if (file_handle_ != nullptr) CloseHandle(static_cast<HANDLE>(file_handle_));
  addr_ = nullptr;
  mapping_handle_ = nullptr;
  file_handle_ = nullptr;
  size_ = 0;
}

MappedFile::MappedFile(MappedFile&& o) noexcept
    : path_(std::move(o.path_)),
      addr_(std::exchange(o.addr_, nullptr)),
      size_(std::exchange(o.size_, 0)),
      file_handle_(std::exchange(o.file_handle_, nullptr)),
      mapping_handle_(std::exchange(o.mapping_handle_, nullptr)) {}

MappedFile& MappedFile::operator=(MappedFile&& o) noexcept {
  if (this != &o) {
    release();
    path_ = std::move(o.path_);
    addr_ = std::exchange(o.addr_, nullptr);
    size_ = std::exchange(o.size_, 0);
    file_handle_ = std::exchange(o.file_handle_, nullptr);
    mapping_handle_ = std::exchange(o.mapping_handle_, nullptr);
  }
  return *this;
}

#else  // POSIX

MappedFile::MappedFile(const std::filesystem::path& path) : path_(path) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) {
    throw Error("cannot open '" + path.string() + "': " + std::strerror(errno));
  }
  struct stat st {};
  if (::fstat(fd_, &st) != 0) {
    const std::string msg = std::strerror(errno);
    release();
    throw Error("cannot stat '" + path.string() + "': " + msg);
  }
  if (!S_ISREG(st.st_mode)) {
    release();
    throw Error("'" + path.string() + "' is not a regular file");
  }
  size_ = static_cast<std::uint64_t>(st.st_size);
  if (size_ == 0) return;  // mmap of length 0 is invalid; nothing to map
  void* p = ::mmap(nullptr, static_cast<size_t>(size_), PROT_READ, MAP_PRIVATE, fd_, 0);
  if (p == MAP_FAILED) {
    const std::string msg = std::strerror(errno);
    release();
    throw Error("cannot mmap '" + path.string() + "': " + msg);
  }
  addr_ = p;
}

void MappedFile::release() noexcept {
  if (addr_ != nullptr) ::munmap(addr_, static_cast<size_t>(size_));
  if (fd_ >= 0) ::close(fd_);
  addr_ = nullptr;
  fd_ = -1;
  size_ = 0;
}

MappedFile::MappedFile(MappedFile&& o) noexcept
    : path_(std::move(o.path_)),
      addr_(std::exchange(o.addr_, nullptr)),
      size_(std::exchange(o.size_, 0)),
      fd_(std::exchange(o.fd_, -1)) {}

MappedFile& MappedFile::operator=(MappedFile&& o) noexcept {
  if (this != &o) {
    release();
    path_ = std::move(o.path_);
    addr_ = std::exchange(o.addr_, nullptr);
    size_ = std::exchange(o.size_, 0);
    fd_ = std::exchange(o.fd_, -1);
  }
  return *this;
}

#endif

MappedFile::~MappedFile() { release(); }

}  // namespace ef
