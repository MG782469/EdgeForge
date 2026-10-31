// EdgeForge - golden reader + comparison helper tests.
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

#include "ef/golden.h"
#include "test_framework.h"

namespace fs = std::filesystem;
using namespace ef;

namespace {

struct TempDir {
  fs::path path;
  TempDir() {
    static std::atomic<int> n{0};
    path = fs::temp_directory_path() / ("edgeforge_golden_" + std::to_string(n++) + "_" +
                                        std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    fs::create_directories(path / "layer0");
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
  void write(const std::string& rel, const void* data, std::size_t bytes) const {
    std::ofstream o(path / rel, std::ios::binary);
    o.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  }
};

}  // namespace

TEST(golden_reads_manifest_and_tensors) {
  TempDir d;
  const std::vector<float> x = {1, 2, 3, 4, 5, 6};
  const std::vector<std::int32_t> ids = {1, 529, 2};
  d.write("layer0/x.f32", x.data(), x.size() * 4);
  d.write("input_ids.i32", ids.data(), ids.size() * 4);
  d.write("bad.f32", x.data(), 5 * 4);
  const std::string manifest = R"({"files": {
    "layer0/x": {"file": "layer0/x.f32", "dtype": "f32", "shape": [2, 3]},
    "input_ids": {"file": "input_ids.i32", "dtype": "i32", "shape": [3]},
    "bad": {"file": "bad.f32", "dtype": "f32", "shape": [2, 3]},
    "gone": {"file": "gone.f32", "dtype": "f32", "shape": [1]}}, "greedy": {"ids": [1]}})";
  d.write("manifest.json", manifest.data(), manifest.size());

  const auto g = GoldenPrompt::open(d.path);
  CHECK(g.has("layer0/x") && !g.has("nope"));
  const auto t = g.load("layer0/x");
  CHECK(t.shape == (std::vector<std::uint64_t>{2, 3}));
  CHECK_EQ(t.row_size(), std::size_t{3});
  CHECK_EQ(t.row(1)[0], 4.0f);
  CHECK(g.input_ids() == ids);
  CHECK_THROWS(g.load("bad"), "shape needs 24");
  CHECK_THROWS(g.load("gone"), "cannot open golden file");
  CHECK_THROWS(g.load("nope"), "not listed");
  CHECK_THROWS(t.row(2), "out of range");
  CHECK_THROWS(GoldenPrompt::open(d.path / "missing"), "cannot open golden manifest");
}

TEST(golden_compare_stats) {
  const std::vector<float> ref = {1, -2, 3, 0.5f};
  auto s = compare(ref.data(), ref.data(), ref.size());
  CHECK(s.max_abs_err == 0.0 && s.rel_err == 0.0 && std::fabs(s.cosine - 1.0) < 1e-12);
  std::vector<float> ours = ref;
  ours[2] = 3.003f;
  s = compare(ours.data(), ref.data(), ref.size());
  CHECK_EQ(s.worst_index, std::size_t{2});
  CHECK(std::fabs(s.rel_err - 0.001) < 1e-6);
  std::vector<float> scaled = {2, -4, 6, 1};  // same direction: cosine 1, large error
  s = compare(scaled.data(), ref.data(), ref.size());
  CHECK(std::fabs(s.cosine - 1.0) < 1e-12 && s.rel_err > 0.9);
  ours[0] = std::numeric_limits<float>::quiet_NaN();
  s = compare(ours.data(), ref.data(), ref.size());
  CHECK_EQ(s.non_finite, std::size_t{1});
}
