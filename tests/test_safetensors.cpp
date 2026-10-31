// EdgeForge — safetensors reader tests: valid files, every rejection path, fuzz.
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "ef/safetensors.h"
#include "test_framework.h"
#include "test_util.h"

using namespace ef;
using ef_test::append_raw;
using ef_test::make_safetensors;

namespace {

std::string entry(const std::string& name, const std::string& dtype, const std::string& shape, std::uint64_t b,
                  std::uint64_t e) {
  return "\"" + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":" + shape + ",\"data_offsets\":[" +
         std::to_string(b) + "," + std::to_string(e) + "]}";
}

std::string pad8(std::string h) {
  while ((8 + h.size()) % 8 != 0) h += ' ';
  return h;
}

std::vector<std::uint8_t> valid_data() {
  std::vector<std::uint8_t> d;
  for (float v : {1.0f, -2.0f, 0.5f, 0.0f, -0.0f, 3.140625f}) append_raw(d, f32_to_bf16(v));  // a: [0,12)
  append_raw(d, 1.5f);                                                                         // b: [12,20)
  append_raw(d, -7.25f);
  for (std::uint16_t h : {0x3C00, 0x7BFF, 0x0001, 0xFC00}) append_raw(d, h);                  // c: [20,28)
  return d;
}

const std::string kHeader = "{\"__metadata__\":{\"format\":\"pt\"}," + entry("a", "BF16", "[2,3]", 0, 12) + "," +
                            entry("b", "F32", "[2]", 12, 20) + "," + entry("c", "F16", "[4]", 20, 28) + "}";

SafetensorsHeader parse_bytes(const std::vector<std::uint8_t>& f) { return parse_safetensors_header(f.data(), f.size()); }

SafetensorsHeader parse_header_with(const std::string& header, const std::vector<std::uint8_t>& data) {
  return parse_bytes(make_safetensors(header, data));
}

std::vector<std::uint8_t> zeros(std::size_t n) { return std::vector<std::uint8_t>(n, 0); }

}  // namespace

TEST(safetensors_valid_header) {
  const auto h = parse_header_with(pad8(kHeader), valid_data());
  CHECK_EQ(h.tensors.size(), std::size_t{3});
  CHECK_EQ(h.tensors[0].name, std::string("a"));  // header order preserved
  CHECK_EQ(h.tensors[2].name, std::string("c"));
  CHECK_EQ(h.metadata.at("format"), std::string("pt"));
  CHECK_EQ(h.data_offset, 8 + h.header_len);
  CHECK_EQ(h.data_offset % 8, std::uint64_t{0});
  CHECK_EQ(h.data_size, std::uint64_t{28});
  CHECK(h.tensors[0].shape == (std::vector<std::uint64_t>{2, 3}));
  CHECK_EQ(h.tensors[0].numel, std::uint64_t{6});
  CHECK(h.tensors[0].dtype == DType::BF16);
  CHECK_EQ(h.tensors[1].begin, std::uint64_t{12});
  CHECK_EQ(h.tensors[1].nbytes(), std::uint64_t{8});
  CHECK_EQ(h.index.at("c"), std::size_t{2});
}

TEST(safetensors_reads_values_from_file) {
  // Unpadded header: tensor data lands at odd addresses, exercising unaligned reads.
  const ef_test::TempFile tf(make_safetensors(kHeader, valid_data()));
  const auto st = SafetensorsFile::open(tf.path());
  CHECK_EQ(st.tensors().size(), std::size_t{3});
  const TensorInfo& a = st.at("a");
  float va[6];
  st.read_f32(a, 0, 6, va);
  CHECK_EQ(va[0], 1.0f);
  CHECK_EQ(va[1], -2.0f);
  CHECK_EQ(va[2], 0.5f);
  CHECK_EQ(va[3], 0.0f);
  CHECK(std::signbit(va[4]));
  CHECK_EQ(va[5], 3.140625f);
  CHECK_EQ(st.raw_bits(a, 1), 0xC000u);
  float vb[2];
  st.read_f32(st.at("b"), 0, 2, vb);
  CHECK_EQ(vb[0], 1.5f);
  CHECK_EQ(vb[1], -7.25f);
  CHECK_EQ(st.raw_bits(st.at("b"), 1), f32_to_bits(-7.25f));
  const TensorInfo& c = st.at("c");
  CHECK_EQ(st.value_f32(c, 0), 1.0f);
  CHECK_EQ(st.value_f32(c, 1), 65504.0f);
  CHECK_EQ(st.value_f32(c, 2), std::ldexp(1.0f, -24));
  CHECK(std::isinf(st.value_f32(c, 3)) && st.value_f32(c, 3) < 0);
  float partial[2];
  st.read_f32(a, 4, 2, partial);  // sub-range read
  CHECK_EQ(partial[1], 3.140625f);
  CHECK_EQ(st.bytes(c).size(), std::size_t{8});
  CHECK(st.find("nope") == nullptr);
  CHECK_THROWS(st.at("model.missing.weight"), "not found");
  CHECK_THROWS(st.at("a.weight"), "closest: a");
  CHECK_THROWS(st.read_f32(a, 5, 2, partial), "outside");
  CHECK_THROWS(st.raw_bits(a, 6), "out of range");
}

TEST(safetensors_rejects_bad_length_field) {
  CHECK_THROWS(parse_bytes({1, 2, 3}), "too small");
  CHECK_THROWS(parse_bytes(make_safetensors("{}", {}, 0)), "header length is 0");
  CHECK_THROWS(parse_bytes(make_safetensors("{}", {}, 100'000'001)), "100 MB");
  CHECK_THROWS(parse_bytes(make_safetensors("{}", {}, 64)), "exceeds the file size");
  CHECK_THROWS(parse_bytes(make_safetensors("{}", {}, -1 + (1LL << 62))), "100 MB");
}

TEST(safetensors_rejects_bad_json) {
  CHECK_THROWS(parse_header_with(" {}", {}), "does not start with '{'");
  CHECK_THROWS(parse_header_with("[]", {}), "does not start with '{'");
  CHECK_THROWS(parse_header_with("{\"a\": {", {}), "invalid header");
  CHECK_THROWS(parse_header_with("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},"
                                 "\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}",
                                 zeros(8)),
               "duplicate key \"a\"");
  CHECK_THROWS(parse_header_with("{\"\\u00e9\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}\x01", zeros(4)),
               "invalid header");
}

TEST(safetensors_rejects_bad_entries) {
  CHECK_THROWS(parse_header_with("{\"a\": 5}", {}), "must be a JSON object");
  CHECK_THROWS(parse_header_with("{\"a\":{\"shape\":[1],\"data_offsets\":[0,4]}}", zeros(4)), "missing \"dtype\"");
  CHECK_THROWS(parse_header_with("{\"a\":{\"dtype\":\"F32\",\"data_offsets\":[0,4]}}", zeros(4)), "missing \"shape\"");
  CHECK_THROWS(parse_header_with("{\"a\":{\"dtype\":\"F32\",\"shape\":[1]}}", zeros(4)), "missing \"data_offsets\"");
  CHECK_THROWS(parse_header_with("{" + entry("a", "Q4", "[1]", 0, 4) + "}", zeros(4)), "unsupported dtype \"Q4\"");
  CHECK_THROWS(parse_header_with("{\"a\":{\"dtype\":7,\"shape\":[1],\"data_offsets\":[0,4]}}", zeros(4)),
               "\"dtype\" must be a string");
  CHECK_THROWS(parse_header_with("{" + entry("a", "F32", "[-1]", 0, 4) + "}", zeros(4)), "non-negative integers");
  CHECK_THROWS(parse_header_with("{" + entry("a", "F32", "[1.5]", 0, 4) + "}", zeros(4)), "non-negative integers");
  CHECK_THROWS(parse_header_with("{" + entry("a", "F32", "4", 0, 4) + "}", zeros(4)), "\"shape\" must be an array");
  CHECK_THROWS(parse_header_with("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4]}}", zeros(4)),
               "exactly 2 integers");
  CHECK_THROWS(parse_header_with("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,\"4\"]}}", zeros(4)),
               "non-negative integers");
  CHECK_THROWS(parse_header_with("{" + entry("a", "F32", "[0]", 8, 4) + "}", zeros(8)), "begin (8) > end (4)");
  CHECK_THROWS(parse_header_with("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4],\"x\":1}}", zeros(4)),
               "unexpected field \"x\"");
  CHECK_THROWS(parse_header_with("{\"__metadata__\":[1]}", {}), "\"__metadata__\" must be a JSON object");
  CHECK_THROWS(parse_header_with("{\"__metadata__\":{\"format\":1}}", {}), "must be a string");
}

TEST(safetensors_rejects_shape_size_mismatch) {
  CHECK_THROWS(parse_header_with("{" + entry("w", "BF16", "[2,3]", 0, 10) + "}", zeros(10)),
               "needs 12 bytes, but data_offsets span 10 bytes");
  CHECK_THROWS(parse_header_with("{" + entry("w", "F32", "[4294967296,4294967296,16]", 0, 4) + "}", zeros(4)),
               "overflows 64 bits");
}

TEST(safetensors_rejects_out_of_bounds_and_truncation) {
  CHECK_THROWS(parse_header_with("{" + entry("w", "BF16", "[2,3]", 0, 12) + "}", zeros(8)),
               "exceeds the data region size 8");
  auto f = make_safetensors(kHeader, valid_data());
  f.resize(f.size() - 4);  // cut the last tensor short
  CHECK_THROWS(parse_bytes(f), "truncated file?");
  auto g = make_safetensors(kHeader, valid_data());
  g.resize(8 + kHeader.size() / 2);  // cut inside the JSON header
  CHECK_THROWS(parse_bytes(g), "exceeds the file size");
}

TEST(safetensors_rejects_overlap_gap_trailing) {
  CHECK_THROWS(parse_header_with("{" + entry("a", "F32", "[3]", 0, 12) + "," + entry("b", "F32", "[2]", 8, 16) + "}",
                                 zeros(16)),
               "overlap");
  CHECK_THROWS(parse_header_with("{" + entry("a", "F32", "[3]", 0, 12) + "," + entry("b", "F32", "[2]", 16, 24) + "}",
                                 zeros(24)),
               "gap of 4 bytes");
  CHECK_THROWS(parse_header_with("{" + entry("a", "F32", "[3]", 0, 12) + "}", zeros(20)), "trailing bytes");
  CHECK_THROWS(parse_header_with("{" + entry("a", "F32", "[3]", 4, 16) + "}", zeros(16)), "gap of 4 bytes");
}

TEST(safetensors_edge_cases_accepted) {
  // zero-sized tensor sharing an offset, scalar tensor, header padded with spaces, unicode name
  const std::string h = pad8("{" + entry("a", "F32", "[1]", 0, 4) + "," + entry("empty", "F32", "[0,7]", 4, 4) + "," +
                             entry("scalar", "I64", "[]", 4, 12) + ",\"\\u00e9\":" +
                             "{\"dtype\":\"U8\",\"shape\":[3],\"data_offsets\":[12,15]}}");
  const auto hd = parse_header_with(h, zeros(15));
  CHECK_EQ(hd.tensors.size(), std::size_t{4});
  CHECK_EQ(hd.tensors[1].numel, std::uint64_t{0});
  CHECK_EQ(hd.tensors[2].numel, std::uint64_t{1});
  CHECK_EQ(hd.tensors[3].name, std::string("\xC3\xA9"));
  const auto empty = parse_header_with("{}", {});
  CHECK_EQ(empty.tensors.size(), std::size_t{0});
  CHECK_EQ(empty.data_size, std::uint64_t{0});
}

TEST(safetensors_file_errors) {
  CHECK_THROWS(SafetensorsFile::open("/definitely/not/here/model.safetensors"), "cannot open");
  const ef_test::TempFile empty(std::vector<std::uint8_t>{});
  CHECK_THROWS(SafetensorsFile::open(empty.path()), "too small");
  const ef_test::TempFile bad(make_safetensors("{\"a\":1}", {}));
  CHECK_THROWS(SafetensorsFile::open(bad.path()), bad.path().filename().string());  // message names the file
}

TEST(safetensors_fuzz_never_crashes) {
  const auto base = make_safetensors(pad8(kHeader), valid_data());
  // Every truncation must be rejected cleanly (only the full file is valid).
  int accepted = 0;
  for (std::size_t len = 0; len < base.size(); ++len) {
    try {
      (void)parse_safetensors_header(base.data(), len);
      ++accepted;
    } catch (const ef::Error&) {
    }
  }
  CHECK_EQ(accepted, 0);
  // Random byte mutations: parse or throw ef::Error, nothing else (run under ASan in CI).
  std::mt19937 rng(2024);
  int ok = 0, rejected = 0;
  for (int iter = 0; iter < 30000; ++iter) {
    auto f = base;
    const int edits = 1 + static_cast<int>(rng() % 3);
    for (int e = 0; e < edits; ++e) {
      const std::size_t pos = rng() % f.size();
      f[pos] = static_cast<std::uint8_t>((rng() % 2) ? rng() % 256 : "{}[]\",:0123456789-"[rng() % 18]);
    }
    try {
      const auto h = parse_safetensors_header(f.data(), f.size());
      for (const auto& t : h.tensors) CHECK(t.end <= h.data_size);  // invariants hold when accepted
      ++ok;
    } catch (const ef::Error&) {
      ++rejected;
    }
  }
  CHECK(rejected > 0);
  CHECK_EQ(ok + rejected, 30000);
}
