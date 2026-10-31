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

namespace ef_test {

// A small hand-made tokenizer.json in the TinyLlama format, for unit tests.
// ids: 0 <unk>, 1 <s>, 2 </s>, 3..258 <0x00>..<0xFF>, then
// 259 "SS", 260 "Sa", 261 "ab", 262 "Sab", 263 "bc", 264 "S", 265 "a", 266 "b", 267 "c",
// 268 "SSS", 269 "h", 270 "i", 271 "aa"   (S = U+2581)
// merge ranks: 0 "b c", 1 "S a", 2 "a b", 3 "Sa b", 4 "S S", 5 "SS S", 6 "a a"
struct MiniTokOptions {
  bool pair_merges = false;
  bool byte_fallback = true;
  bool with_config = true;        // tokenizer_config.json with legacy=false
  std::string pre_tokenizer = "null";
};

inline std::string mini_tokenizer_json(const MiniTokOptions& o = {}) {
  const std::string S = "\xE2\x96\x81";
  std::string vocab = "\"<unk>\":0,\"<s>\":1,\"</s>\":2";
  static const char* hex = "0123456789ABCDEF";
  for (int b = 0; b < 256; ++b) {
    vocab += std::string(",\"<0x") + hex[b >> 4] + hex[b & 15] + ">\":" + std::to_string(3 + b);
  }
  const std::pair<std::string, int> extra[] = {{S + S, 259}, {S + "a", 260}, {"ab", 261}, {S + "ab", 262},
                                               {"bc", 263},  {S, 264},       {"a", 265},  {"b", 266},
                                               {"c", 267},   {S + S + S, 268}, {"h", 269}, {"i", 270}, {"aa", 271}};
  for (const auto& [p, id] : extra) vocab += ",\"" + p + "\":" + std::to_string(id);
  const std::pair<std::string, std::string> merges[] = {{"b", "c"}, {S, "a"}, {"a", "b"}, {S + "a", "b"},
                                                         {S, S},     {S + S, S}, {"a", "a"}};
  std::string m;
  for (const auto& [x, y] : merges) {
    if (!m.empty()) m += ",";
    m += o.pair_merges ? "[\"" + x + "\",\"" + y + "\"]" : "\"" + x + " " + y + "\"";
  }
  return std::string("{\"version\":\"1.0\",\"added_tokens\":[") +
         "{\"id\":0,\"content\":\"<unk>\",\"special\":true},{\"id\":1,\"content\":\"<s>\",\"special\":true}," +
         "{\"id\":2,\"content\":\"</s>\",\"special\":true}]," +
         "\"normalizer\":{\"type\":\"Sequence\",\"normalizers\":[{\"type\":\"Prepend\",\"prepend\":\"" + S +
         "\"},{\"type\":\"Replace\",\"pattern\":{\"String\":\" \"},\"content\":\"" + S + "\"}]}," +
         "\"pre_tokenizer\":" + o.pre_tokenizer + "," +
         "\"model\":{\"type\":\"BPE\",\"unk_token\":\"<unk>\",\"fuse_unk\":true,\"byte_fallback\":" +
         (o.byte_fallback ? "true" : "false") + ",\"vocab\":{" + vocab + "},\"merges\":[" + m + "]}}";
}

inline std::string mini_tokenizer_config() {
  return "{\"legacy\":false,\"bos_token\":\"<s>\",\"eos_token\":{\"content\":\"</s>\"}}";
}

}  // namespace ef_test
