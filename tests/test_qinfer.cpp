// EdgeForge - Stage 6 tests: packed-weight kernels and the engine running from an .efq file.
#include <atomic>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "ef/eval_quant.h"
#include "ef/fp16.h"
#include "ef/golden.h"
#include "ef/kernels.h"
#include "ef/model.h"
#include "ef/qkernels.h"
#include "ef/quant.h"
#include "ef/quantizer.h"
#include "ef/threadpool.h"
#include "engine_golden.h"
#include "test_framework.h"
#include "tiny_model.h"

using namespace ef;

namespace {

// Random packed rows: arbitrary codes and finite FP16 scales of both signs (not only what the
// quantizer would produce), so the kernels are tested on every bit pattern they can meet.
std::vector<std::uint8_t> random_rows(QType t, std::size_t rows, std::size_t cols, std::mt19937& rng) {
  const std::size_t rb = static_cast<std::size_t>(qtype_row_bytes(t, cols));
  std::vector<std::uint8_t> v(rows * rb);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_real_distribution<float> scale(-0.05f, 0.05f);
  const std::size_t bsz = t == QType::Q4_0 ? sizeof(BlockQ4_0) : sizeof(BlockQ8_0);
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<std::uint8_t>(byte(rng));
  for (std::size_t off = 0; off < v.size(); off += bsz) {
    const std::uint16_t h = f32_to_fp16(scale(rng));
    std::memcpy(v.data() + off, &h, 2);
  }
  return v;
}

// Runs a test section with a given kernel backend and restores the previous one.
struct BackendScope {
  kern::Backend old = kern::backend();
  explicit BackendScope(kern::Backend b) { kern::set_backend(b); }
  ~BackendScope() { kern::set_backend(old); }
  BackendScope(const BackendScope&) = delete;
  BackendScope& operator=(const BackendScope&) = delete;
};

std::vector<kern::Backend> all_backends() {
  std::vector<kern::Backend> v = {kern::Backend::Scalar};
  if (kern::avx2_compiled()) v.push_back(kern::Backend::Avx2);
  return v;
}

// |kernel - exact| relative to sum |w_i x_i| (the scale of rounding error in any summation order).
double rel_dot_error(QType t, const std::uint8_t* row, const float* x, std::size_t n, float got) {
  std::vector<float> w(n);
  dequantize_row(t, row, n, w.data());
  double exact = 0.0, mag = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    exact += static_cast<double>(w[i]) * static_cast<double>(x[i]);
    mag += std::fabs(static_cast<double>(w[i]) * static_cast<double>(x[i]));
  }
  return std::fabs(static_cast<double>(got) - exact) / std::max(mag, 1e-30);
}

}  // namespace

TEST(qinfer_packed_dots_match_dequantized_weights) {
  std::mt19937 rng(61);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  double worst4 = 0.0, worst8 = 0.0;
  for (std::size_t n : {std::size_t{32}, std::size_t{64}, std::size_t{2048}, std::size_t{5632}}) {
    std::vector<float> x(n);
    for (auto& v : x) v = nd(rng) * (nd(rng) > 2.0f ? 30.0f : 1.0f);  // a few outlier channels
    for (QType t : {QType::Q4_0, QType::Q8_0}) {
      const std::size_t rows = 24, rb = static_cast<std::size_t>(qtype_row_bytes(t, n));
      const auto W = random_rows(t, rows, n, rng);
      std::vector<float> y(rows);
      if (t == QType::Q4_0) {
        kern::matvec_q4_0(y.data(), W.data(), x.data(), rows, n);
      } else {
        kern::matvec_q8_0(y.data(), W.data(), x.data(), rows, n);
      }
      for (std::size_t r = 0; r < rows; ++r) {
        const float d = t == QType::Q4_0 ? kern::dot_q4_0(W.data() + r * rb, x.data(), n)
                                         : kern::dot_q8_0(W.data() + r * rb, x.data(), n);
        CHECK(f32_to_bits(d) == f32_to_bits(y[r]));  // matvec = row dots, bit for bit
        const double e = rel_dot_error(t, W.data() + r * rb, x.data(), n, y[r]);
        (t == QType::Q4_0 ? worst4 : worst8) = std::max(t == QType::Q4_0 ? worst4 : worst8, e);
      }
    }
  }
  CHECK(worst4 < 1e-5 && worst8 < 1e-5);
  std::printf("    packed vs dequantized-then-dot (relative to sum|w*x|): Q4_0 %.2e, Q8_0 %.2e\n", worst4, worst8);
}

TEST(qinfer_packed_dot_layout_and_edge_codes) {
  // One Q4_0 block by hand: code c in byte j's low nibble is weight j, high nibble weight j+16.
  std::uint8_t blk[18] = {};
  const std::uint16_t one = f32_to_fp16(1.0f);
  std::memcpy(blk, &one, 2);
  for (std::size_t j = 0; j < 16; ++j) blk[2 + j] = 0x88;  // all codes 8 -> weight 0
  blk[2 + 3] = 0x0F;                                         // weight 3 = 15-8 = +7, weight 19 = 0-8 = -8
  float x[32] = {};
  x[3] = 2.0f;
  x[19] = 0.5f;
  CHECK(kern::dot_q4_0(blk, x, 32) == 7.0f * 2.0f + -8.0f * 0.5f);
  const std::uint16_t zero = 0;
  std::memcpy(blk, &zero, 2);  // zero scale: block contributes exactly 0
  CHECK(kern::dot_q4_0(blk, x, 32) == 0.0f);
  std::uint8_t b8[34] = {};
  std::memcpy(b8, &one, 2);
  b8[2 + 0] = static_cast<std::uint8_t>(-127);
  b8[2 + 31] = 127;
  for (auto& v : x) v = 1.0f;
  CHECK(kern::dot_q8_0(b8, x, 32) == 0.0f);
  b8[2 + 5] = 3;
  CHECK(kern::dot_q8_0(b8, x, 32) == 3.0f);
}

TEST(qinfer_dispatch_every_type_and_row_lookup) {
  const BackendScope scalar(kern::Backend::Scalar);  // bit-identity with the Stage 2 kernels below
  std::mt19937 rng(62);
  std::normal_distribution<float> nd(0.0f, 0.3f);
  const std::size_t rows = 7, cols = 96;
  std::vector<float> src(rows * cols), x(cols);
  for (auto& v : src) v = nd(rng);
  for (auto& v : x) v = nd(rng);
  for (QType t : {QType::F32, QType::F16, QType::BF16, QType::Q8_0, QType::Q4_0}) {
    const std::size_t rb = static_cast<std::size_t>(qtype_row_bytes(t, cols));
    std::vector<std::uint8_t> W(rows * rb);
    for (std::size_t r = 0; r < rows; ++r) quantize_row(t, src.data() + r * cols, cols, W.data() + r * rb);
    const WeightView view{t, W.data(), rows, cols};
    CHECK_EQ(view.bytes(), static_cast<std::uint64_t>(W.size()));
    std::vector<float> y(rows), deq(cols), row(cols);
    kern::matvec(y.data(), view, x.data());
    for (std::size_t r = 0; r < rows; ++r) {
      CHECK(rel_dot_error(t, W.data() + r * rb, x.data(), cols, y[r]) < 1e-5);
      dequantize_row(t, W.data() + r * rb, cols, deq.data());
      kern::row_f32(row.data(), view, r);
      CHECK(row == deq);  // embedding lookup = exact dequantization of that row
    }
    if (t == QType::BF16) {  // the reference engine's kernels, bit for bit
      std::vector<float> ref(rows), emb(cols);
      kern::matvec_bf16(ref.data(), reinterpret_cast<const std::uint16_t*>(W.data()), x.data(), rows, cols);
      CHECK(ref == y);
      kern::embedding_bf16(emb.data(), reinterpret_cast<const std::uint16_t*>(W.data()), 4, cols);
      kern::row_f32(row.data(), view, 4);
      CHECK(emb == row);
    }
    if (t == QType::F32) {
      std::vector<float> ref(rows);
      kern::matvec_f32(ref.data(), reinterpret_cast<const float*>(W.data()), x.data(), rows, cols);
      CHECK(ref == y);
    }
  }
}

// ---------------------------------------------------------------- Stage 7: AVX2 + threads

TEST(qinfer_avx2_kernels_match_the_exact_dot) {
  if (!kern::avx2_compiled()) {
    std::printf("    (skipped: built without AVX2)\n");
    return;
  }
#if defined(EF_HAVE_AVX2)
  std::mt19937 rng(63);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  double worst = 0.0;
  for (std::size_t n : {std::size_t{32}, std::size_t{64}, std::size_t{96}, std::size_t{2048}, std::size_t{5632}}) {
    std::vector<float> x(n);
    for (auto& v : x) v = nd(rng) * (nd(rng) > 2.0f ? 30.0f : 1.0f);
    for (QType t : {QType::Q4_0, QType::Q8_0}) {
      const auto W = random_rows(t, 8, n, rng);
      const std::size_t rb = static_cast<std::size_t>(qtype_row_bytes(t, n));
      for (std::size_t r = 0; r < 8; ++r) {
        const std::uint8_t* row = W.data() + r * rb;
        const float got = t == QType::Q4_0 ? kern::dot_q4_0_avx2(row, x.data(), n) : kern::dot_q8_0_avx2(row, x.data(), n);
        worst = std::max(worst, rel_dot_error(t, row, x.data(), n, got));
      }
    }
  }
  for (std::size_t n : {std::size_t{1}, std::size_t{15}, std::size_t{33}, std::size_t{100}, std::size_t{2048}}) {
    std::vector<float> w(n), x(n);
    for (auto& v : w) v = nd(rng);
    for (auto& v : x) v = nd(rng);
    std::vector<std::uint16_t> wb(n);
    for (std::size_t i = 0; i < n; ++i) wb[i] = f32_to_bf16(w[i]);
    const float gb = kern::dot_bf16_avx2(wb.data(), x.data(), n), gf = kern::dot_f32_avx2(w.data(), x.data(), n);
    worst = std::max(worst, rel_dot_error(QType::BF16, reinterpret_cast<const std::uint8_t*>(wb.data()), x.data(), n, gb));
    worst = std::max(worst, rel_dot_error(QType::F32, reinterpret_cast<const std::uint8_t*>(w.data()), x.data(), n, gf));
  }
  CHECK(worst < 1e-5);
  std::printf("    AVX2 kernels vs exact dot (relative to sum|w*x|): worst %.2e\n", worst);
#endif
}

TEST(qinfer_threadpool_covers_every_index_once) {
  for (unsigned T : {1u, 2u, 3u, 8u}) {
    ThreadPool pool(T);
    CHECK_EQ(pool.size(), T);
    for (int rep = 0; rep < 300; ++rep) {
      for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{5}, std::size_t{100}, std::size_t{1001}}) {
        std::vector<std::atomic<int>> hits(n);
        std::atomic<unsigned> max_chunk{0};
        pool.parallel_for(
            n,
            [&](std::size_t c, std::size_t b, std::size_t e) {
              unsigned prev = max_chunk.load();
              while (c > prev && !max_chunk.compare_exchange_weak(prev, static_cast<unsigned>(c))) {
              }
              for (std::size_t i = b; i < e; ++i) hits[i].fetch_add(1);
            },
            static_cast<std::size_t>(rep % 3 == 0 ? 1 : 16));
        int bad = 0;
        for (auto& h : hits) bad += h.load() != 1;
        CHECK_EQ(bad, 0);
        CHECK(max_chunk.load() < T);
      }
    }
  }
}

TEST(qinfer_parallel_gemv_is_bit_identical_for_any_thread_count) {
  std::mt19937 rng(64);
  std::normal_distribution<float> nd(0.0f, 0.3f);
  const std::size_t cols = 256;
  std::vector<float> x(cols);
  for (auto& v : x) v = nd(rng);
  std::vector<std::vector<std::uint8_t>> store;
  std::vector<WeightView> views;
  for (auto [t, rows] : {std::pair{QType::Q4_0, std::size_t{300}}, std::pair{QType::Q8_0, std::size_t{37}},
                         std::pair{QType::BF16, std::size_t{70}}, std::pair{QType::F32, std::size_t{5}}}) {
    std::vector<float> src(rows * cols);
    for (auto& v : src) v = nd(rng);
    const std::size_t rb = static_cast<std::size_t>(qtype_row_bytes(t, cols));
    store.emplace_back(rows * rb);
    for (std::size_t r = 0; r < rows; ++r) quantize_row(t, src.data() + r * cols, cols, store.back().data() + r * rb);
    views.push_back(WeightView{t, store.back().data(), rows, cols});
  }
  for (kern::Backend b : all_backends()) {
    const BackendScope scope(b);
    auto run_all = [&](ThreadPool* pool) {
      std::vector<std::vector<float>> ys;
      for (const auto& v : views) ys.emplace_back(v.rows);
      std::vector<kern::MatvecJob> jobs;
      for (std::size_t i = 0; i < views.size(); ++i) jobs.push_back({ys[i].data(), &views[i], x.data()});
      kern::matvec_jobs(jobs.data(), jobs.size(), pool);
      return ys;
    };
    const auto ref = run_all(nullptr);
    for (unsigned T : {2u, 3u, 5u, 8u}) {
      ThreadPool pool(T);
      CHECK(run_all(&pool) == ref);
      std::vector<float> one(views[0].rows);
      kern::matvec(one.data(), views[0], x.data(), &pool);
      CHECK(one == ref[0]);
    }
  }
}

// ---------------------------------------------------------------- the engine on an .efq file

namespace {

ef_test::Tiny tiny32() {  // rows divisible by 32 so Q4_0 / Q8_0 apply everywhere
  ef_test::Tiny t;
  t.D = 32;
  t.H = 4;
  t.KV = 2;
  t.hd = 8;
  t.F = 64;
  t.V = 40;
  return t;
}

// Logits after each token of `ids` (one position at a time through the KV cache).
std::vector<std::vector<float>> run(const Model& m, const std::vector<std::int32_t>& ids) {
  Transformer tr(m, ids.size());
  std::vector<std::vector<float>> out;
  const auto V = static_cast<std::size_t>(m.config().vocab_size);
  for (std::size_t p = 0; p < ids.size(); ++p) {
    const float* lg = tr.forward(ids[p], p);
    out.emplace_back(lg, lg + V);
  }
  return out;
}

const std::vector<std::int32_t> kIds = {1, 5, 9, 33, 2, 17, 17, 4, 30, 12};

}  // namespace

TEST(qinfer_lossless_efq_engine_is_bit_identical_to_safetensors) {
  for (bool tied : {false, true}) {
    ef_test::Tiny t = tiny32();
    t.tied = tied;
    const ef_test::TinyDir d(t, 71);
    QuantizeOptions o;
    o.layers = QType::BF16;
    o.lm_head = QType::BF16;
    quantize_model(d.path, d.path / "bf16.efq", o);
    const Model ref = Model::load(d.path);
    const Model q = Model::load_efq(d.path / "bf16.efq");
    CHECK(q.lm_head().data == q.embed_tokens().data || !tied);
    CHECK(q.layer(1).w_up.type == QType::BF16 && q.description().rfind("efq", 0) == 0);
    CHECK(run(ref, kIds) == run(q, kIds));  // same weights, same kernels: identical logits
  }
}

TEST(qinfer_quantized_engine_matches_its_dequantized_reference) {
  const ef_test::TinyDir d(tiny32(), 72);
  QuantizeOptions o;  // uniform Q4_0 layers, Q8_0 lm_head, BF16 embedding
  o.overrides = parse_overrides("model.layers.1.mlp.down_proj.weight=q8_0");  // both packed kernels in layers
  quantize_model(d.path, d.path / "q.efq", o);
  const Model q = Model::load_efq(d.path / "q.efq");
  CHECK(q.layer(0).wq.type == QType::Q4_0 && q.layer(1).w_down.type == QType::Q8_0 && q.lm_head().type == QType::Q8_0);
  CHECK(q.description().find("13 Q4_0 + 2 Q8_0") != std::string::npos);
  const Model deq = Model::dequantized_copy(q);
  CHECK(deq.layer(0).wq.type == QType::F32);
  const auto a = run(q, kIds), b = run(deq, kIds), src = run(Model::load(d.path), kIds);
  double worst_rel = 0.0, worst_cos = 1.0, src_cos = 1.0;
  std::size_t argmax_same = 0;
  for (std::size_t p = 0; p < a.size(); ++p) {
    const DiffStats s = compare(a[p].data(), b[p].data(), a[p].size());
    worst_rel = std::max(worst_rel, s.rel_err);
    worst_cos = std::min(worst_cos, s.cosine);
    argmax_same += kern::argmax(a[p].data(), a[p].size()) == kern::argmax(b[p].data(), b[p].size());
    src_cos = std::min(src_cos, compare(a[p].data(), src[p].data(), a[p].size()).cosine);
  }
  // Packed kernels only change the summation order vs the dequantized weights.
  CHECK(worst_rel < 1e-4 && worst_cos > 1.0 - 1e-9);
  CHECK_EQ(argmax_same, a.size());
  // ...and the weights really are quantized (a random 32-wide toy net amplifies Q4 noise a lot,
  // so this only rules out garbage; real-model quality is measured by eval-quant).
  CHECK(src_cos > 0.5 && src_cos < 1.0 - 1e-6);
  std::printf("    Q engine vs dequantized reference: max rel err %.2e; vs BF16 source: min cosine %.6f\n", worst_rel,
              src_cos);
  // Greedy decoding through the KV cache agrees too.
  Transformer t1(q, 24), t2(deq, 24);
  CHECK(generate_greedy(t1, {1, 5, 9}, 12, -1).tokens == generate_greedy(t2, {1, 5, 9}, 12, -1).tokens);
}

// ---------------------------------------------------------------- eval-quant

TEST(qinfer_eval_quant_is_exact_on_a_lossless_file) {
  const ef_test::TinyDir d(tiny32(), 81);
  const auto golden = d.path / "golden";
  ef_test::write_engine_golden(d.path, golden / "prompts" / "p0_full", {1, 5, 9, 33, 2, 17, 4, 30}, true, 8);
  ef_test::write_engine_golden(d.path, golden / "prompts" / "p1_last", {1, 12, 7}, false, 8);
  QuantizeOptions o;
  o.layers = QType::BF16;
  o.lm_head = QType::BF16;
  quantize_model(d.path, d.path / "bf16.efq", o);
  EvalQuantReport r;
  EvalQuantOptions eo;
  eo.qmodel = d.path / "bf16.efq";
  eo.model_dir = d.path;
  eo.golden_dir = golden;
  eo.text_ids = {1, 3, 8, 21, 22, 5, 6, 39, 11, 2, 13, 14};
  eo.report = &r;
  CHECK_EQ(run_eval_quant(eo), 0);
  // Golden: p0 8 prompt positions + 2 decode steps, p1 last position + 2 decode steps; text: 11.
  CHECK_EQ(r.golden_positions, std::size_t{8 + 2 + 1 + 2});
  CHECK_EQ(r.text_positions, std::size_t{11});
  CHECK(r.kl_mean == 0.0 && r.kl_max == 0.0 && r.top1 == r.positions);  // identical logits
  CHECK(r.ppl_ref > 1.0 && r.ppl_q == r.ppl_ref);
  CHECK(r.greedy_prompts == 2 && r.greedy_identical == 2);
  CHECK_EQ(r.kernel_rows, std::size_t{0});  // nothing packed in this file
  CHECK(r.tps_q > 0.0 && r.tps_ref > 0.0 && r.q_peak_rss > 0);
}

TEST(qinfer_eval_quant_measures_a_quantized_file) {
  const ef_test::TinyDir d(tiny32(), 82);
  const auto golden = d.path / "golden";
  ef_test::write_engine_golden(d.path, golden / "prompts" / "p0_full", {1, 5, 9, 33, 2, 17, 4, 30}, true, 8);
  ef_test::write_engine_golden(d.path, golden / "prompts" / "p1_last", {1, 12, 7}, false, 8);
  quantize_model(d.path, d.path / "q.efq", QuantizeOptions{});
  EvalQuantReport r;
  EvalQuantOptions eo;
  eo.qmodel = d.path / "q.efq";
  eo.model_dir = d.path;
  eo.golden_dir = golden;
  eo.text_ids = {1, 3, 8, 21, 22, 5, 6, 39, 11, 2, 13, 14};
  eo.report = &r;
  run_eval_quant(eo);  // quality targets are for the real model; a random toy net need not meet them
  CHECK(r.kernel_rows == 15 * 16 && r.kernel_max_rel < 1e-5);  // 14 layer matrices + lm_head, 16 rows each
  CHECK_EQ(r.positions, std::size_t{13 + 11});
  CHECK(r.nonfinite == 0 && r.kl_mean > 0.0 && r.kl_max >= r.kl_mean);
  CHECK(r.ppl_ref > 1.0 && r.ppl_q > 1.0 && r.ppl_q != r.ppl_ref);
  CHECK_EQ(r.greedy_prompts, std::size_t{2});
  // The check list: kernels, NaN/Inf, mean KL, confident top-1, perplexity ratio.
  CHECK_EQ(r.passed + r.failed, 5);
}

TEST(qinfer_engine_threads_and_backends) {
  const ef_test::TinyDir d(tiny32(), 73);
  quantize_model(d.path, d.path / "q.efq", QuantizeOptions{});
  const Model q = Model::load_efq(d.path / "q.efq");
  auto run_pool = [&](ThreadPool* pool) {
    Transformer tr(q, kIds.size(), pool);
    std::vector<std::vector<float>> out;
    const auto V = static_cast<std::size_t>(q.config().vocab_size);
    for (std::size_t p = 0; p < kIds.size(); ++p) {
      const float* lg = tr.forward(kIds[p], p);
      out.emplace_back(lg, lg + V);
    }
    return out;
  };
  std::vector<std::vector<std::vector<float>>> per_backend;
  for (kern::Backend b : all_backends()) {
    const BackendScope scope(b);
    const auto ref = run_pool(nullptr);
    for (unsigned T : {2u, 3u, 7u}) {
      ThreadPool pool(T);
      CHECK(run_pool(&pool) == ref);  // same logits, bit for bit, for any thread count
    }
    per_backend.push_back(ref);
  }
  if (per_backend.size() == 2) {  // AVX2 vs scalar: only the rounding order differs
    double worst = 0.0;
    std::size_t same = 0;
    for (std::size_t p = 0; p < kIds.size(); ++p) {
      const DiffStats s = compare(per_backend[1][p].data(), per_backend[0][p].data(), per_backend[0][p].size());
      worst = std::max(worst, s.rel_err);
      same += kern::argmax(per_backend[1][p].data(), per_backend[1][p].size()) ==
              kern::argmax(per_backend[0][p].data(), per_backend[0][p].size());
    }
    CHECK(worst < 1e-4);
    CHECK_EQ(same, kIds.size());
    std::printf("    engine AVX2 vs scalar logits: max rel err %.2e\n", worst);
  }
}
