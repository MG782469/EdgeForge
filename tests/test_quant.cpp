// EdgeForge - Stage 5 tests: Q4_0/Q8_0 blocks, the .efq writer/reader, determinism, corruption.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "ef/fp16.h"
#include "ef/json.h"
#include "ef/kernels.h"
#include "ef/model.h"
#include "ef/model_config.h"
#include "ef/qfile.h"
#include "ef/quant.h"
#include "ef/quantizer.h"
#include "ef/verify_quant.h"
#include "test_framework.h"
#include "test_util.h"
#include "tiny_model.h"

using namespace ef;

namespace {

double block_sse(const float* x, const float* y) {
  double s = 0;
  for (std::size_t j = 0; j < kQK; ++j) s += (double(x[j]) - double(y[j])) * (double(x[j]) - double(y[j]));
  return s;
}

std::vector<std::uint8_t> slurp(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

ef_test::Tiny tiny32() {  // row lengths multiple of 32 so Q4/Q8 apply
  ef_test::Tiny t;
  t.D = 32;
  t.H = 4;
  t.KV = 2;
  t.hd = 8;
  t.F = 64;
  return t;
}

}  // namespace

TEST(quant_q4_exact_on_grid_and_nibble_layout) {
  float x[kQK], y[kQK];
  for (std::size_t j = 0; j < kQK; ++j) x[j] = static_cast<float>(static_cast<int>(j % 16) - 8) * 0.25f;
  BlockQ4_0 b;
  quantize_block_q4_0(x, b);
  dequantize_block_q4_0(b, y);
  CHECK(std::memcmp(x, y, sizeof x) == 0);  // values on the grid round-trip exactly
  BlockQ4_0 m{};
  m.d = f32_to_fp16(1.0f);
  m.qs[0] = 0x2F;  // low nibble 15 -> w[0] = +7, high nibble 2 -> w[16] = -6
  dequantize_block_q4_0(m, y);
  CHECK(y[0] == 7.0f && y[16] == -6.0f && y[1] == -8.0f);
  const float zeros[kQK] = {};
  quantize_block_q4_0(zeros, b);
  CHECK(b.d == 0 && b.qs[0] == 0x88);
  dequantize_block_q4_0(b, y);
  CHECK(y[5] == 0.0f);
}

TEST(quant_q4_error_bound_and_search_never_worse) {
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.0f, 0.02f);
  double sse_search = 0, sse_plain = 0, ref = 0;
  int worse = 0, out_of_bound = 0;
  for (int k = 0; k < 20000; ++k) {
    float x[kQK], a[kQK], p[kQK];
    for (auto& v : x) v = nd(rng);
    if (k % 7 == 0) x[k % kQK] *= 6.0f;  // outliers
    BlockQ4_0 bs, bp;
    quantize_block_q4_0(x, bs, true);
    quantize_block_q4_0(x, bp, false);
    dequantize_block_q4_0(bs, a);
    dequantize_block_q4_0(bp, p);
    const double es = block_sse(x, a), ep = block_sse(x, p);
    worse += es > ep * (1 + 1e-12);
    sse_search += es;
    sse_plain += ep;
    // absmax mode never clips beyond one step (FP16 scale rounding aside); MSE search may clip
    // a lone outlier harder in exchange for a lower total error, so only its SSE is checked.
    const float d = std::fabs(fp16_to_f32(bp.d));
    for (std::size_t j = 0; j < kQK; ++j) out_of_bound += std::fabs(x[j] - p[j]) > d * 1.01f;
    for (auto v : x) ref += double(v) * double(v);
  }
  CHECK_EQ(worse, 0);
  CHECK_EQ(out_of_bound, 0);  // |error| <= one step, even for clipped values
  const double rel_s = std::sqrt(sse_search / ref), rel_p = std::sqrt(sse_plain / ref);
  CHECK(rel_s < rel_p && rel_s < 0.11);
  std::printf("    Q4_0 rel RMSE on N(0,1)+outliers: absmax %.4f, MSE search %.4f\n", rel_p, rel_s);
}

TEST(quant_q8_error_bound) {
  std::mt19937 rng(12);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  double sse = 0, ref = 0;
  int bad = 0;
  for (int k = 0; k < 5000; ++k) {
    float x[kQK], y[kQK];
    for (auto& v : x) v = nd(rng);
    BlockQ8_0 b;
    quantize_block_q8_0(x, b);
    dequantize_block_q8_0(b, y);
    const float d = fp16_to_f32(b.d);
    for (std::size_t j = 0; j < kQK; ++j) {
      bad += std::fabs(x[j] - y[j]) > d * 0.5001f + 1e-7f;
      const double e = double(x[j]) - double(y[j]);
      sse += e * e;
      ref += double(x[j]) * double(x[j]);
    }
  }
  CHECK_EQ(bad, 0);
  CHECK(std::sqrt(sse / ref) < 0.01);
}

TEST(quant_rows_types_and_fake_group) {
  CHECK_EQ(qtype_row_bytes(QType::Q4_0, 2048), std::uint64_t{64 * 18});
  CHECK_EQ(qtype_row_bytes(QType::Q8_0, 2048), std::uint64_t{64 * 34});
  CHECK_EQ(qtype_row_bytes(QType::BF16, 2048), std::uint64_t{4096});
  CHECK_THROWS(qtype_row_bytes(QType::Q4_0, 30), "divisible by 32");
  CHECK(qtype_from_string("q4_0") == QType::Q4_0 && qtype_from_string("Q8_0") == QType::Q8_0);
  CHECK(!qtype_from_string("q3_k"));
  CHECK(std::fabs(qtype_bits_per_weight(QType::Q4_0) - 4.5) < 1e-12);
  std::mt19937 rng(13);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> x(256), y(256), z(256);
  for (auto& v : x) v = nd(rng);
  for (QType t : {QType::F32, QType::F16, QType::BF16, QType::Q8_0, QType::Q4_0}) {
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(qtype_row_bytes(t, 256)));
    quantize_row(t, x.data(), 256, buf.data());
    dequantize_row(t, buf.data(), 256, y.data());
    double e = 0;
    for (std::size_t i = 0; i < 256; ++i) e = std::max(e, double(std::fabs(x[i] - y[i])));
    CHECK(t != QType::F32 || e == 0.0);
    CHECK(e < 0.6);
  }
  for (std::size_t b = 0; b < 256; b += kQK) {  // group-32 fake quant == the real Q4_0 block
    BlockQ4_0 blk;
    quantize_block_q4_0(x.data() + b, blk);
    dequantize_block_q4_0(blk, y.data() + b);
    fake_quant_q4_group(x.data() + b, kQK, z.data() + b, true);
  }
  CHECK(y == z);
}

TEST(qfile_roundtrip_matches_source_and_is_deterministic) {
  const ef_test::Tiny t = tiny32();
  const ef_test::TinyDir d(t, 21);
  const auto f1 = d.path / "m1.efq", f2 = d.path / "m2.efq";
  QuantizeOptions o;
  o.threads = 3;
  const QuantizeResult r = quantize_model(d.path, f1, o);
  o.threads = 1;
  quantize_model(d.path, f2, o);
  CHECK(slurp(f1) == slurp(f2));  // byte-identical regardless of thread count
  const QuantFile q = QuantFile::open(f1);
  CHECK_EQ(q.file_size(), r.file_bytes);
  CHECK_EQ(q.bytes_per_token(), r.bytes_per_token);
  CHECK_EQ(q.tensors().size(), std::size_t{21});
  CHECK(q.at("model.embed_tokens.weight").type == QType::BF16);
  CHECK(q.at("model.norm.weight").type == QType::F32);
  CHECK(q.at("model.layers.1.mlp.down_proj.weight").type == QType::Q4_0);
  CHECK(q.at("lm_head.weight").type == QType::Q8_0);
  CHECK_EQ(q.tensors().front().name, std::string("model.embed_tokens.weight"));  // forward-pass order
  const SafetensorsFile st = SafetensorsFile::open(d.path / "model.safetensors");
  int bad_offsets = 0, bad_rows = 0;
  for (const auto& qt : q.tensors()) {
    bad_offsets += qt.offset % 64 != 0;
    const std::uint64_t rb = qtype_row_bytes(qt.type, qt.cols());
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(rb * qt.rows()));
    quantize_tensor_rows(st, st.at(qt.name), qt.type, true, 0, qt.rows(), buf.data());
    bad_rows += std::memcmp(buf.data(), q.data(qt), buf.size()) != 0;
  }
  CHECK_EQ(bad_offsets, 0);
  CHECK_EQ(bad_rows, 0);
  std::vector<float> row(32), src(32);
  q.dequantize_row(q.at("model.embed_tokens.weight"), 3, row.data());
  st.read_f32(st.at("model.embed_tokens.weight"), 3 * 32, 32, src.data());
  CHECK(row == src);  // BF16 embedding stored losslessly
  const double rel = q.at("model.layers.0.self_attn.q_proj.weight").rel_rmse;
  CHECK(rel > 0.01 && rel < 0.12);
  CHECK_THROWS(q.row_data(q.at("lm_head.weight"), 11), "out of range");
  VerifyQuantOptions vo;
  vo.qmodel = f1;
  vo.model_dir = d.path;
  CHECK_EQ(run_verify_quant(vo), 0);
}

TEST(qfile_falls_back_to_bf16_for_odd_rows) {
  const ef_test::Tiny t;  // D = 16: rows not divisible by 32
  const ef_test::TinyDir d(t, 22);
  const QuantizeResult r = quantize_model(d.path, d.path / "m.efq", QuantizeOptions{});
  CHECK(!r.notes.empty());
  const QuantFile q = QuantFile::open(d.path / "m.efq");
  CHECK(q.at("model.layers.0.self_attn.q_proj.weight").type == QType::BF16);
  CHECK_EQ(q.at("model.layers.0.self_attn.q_proj.weight").rel_rmse, 0.0);
}

TEST(qfile_rejects_corruption) {
  const ef_test::Tiny t = tiny32();
  const ef_test::TinyDir d(t, 23);
  quantize_model(d.path, d.path / "m.efq", QuantizeOptions{});
  const auto good = slurp(d.path / "m.efq");
  CHECK_EQ(parse_efq(good.data(), good.size()).tensors.size(), std::size_t{21});
  auto bad = good;
  bad[0] = 'X';
  CHECK_THROWS(parse_efq(bad.data(), bad.size()), "bad magic");
  bad = good;
  bad[4] = 2;
  CHECK_THROWS(parse_efq(bad.data(), bad.size()), "unsupported .efq version 2");
  CHECK_THROWS(parse_efq(good.data(), good.size() - 1), "past the end");
  CHECK_THROWS(parse_efq(good.data(), 10), "too small");
  bad = good;  // nudge the first tensor offset by one byte (same text length)
  std::string text(bad.begin() + 16, bad.begin() + 16 + 4096);
  const auto pos = text.find("\"offset\":") + 9;
  auto& digit = bad[16 + pos];
  digit = digit == '9' ? '8' : static_cast<std::uint8_t>(digit + 1);
  CHECK_THROWS(parse_efq(bad.data(), bad.size()), "64-byte aligned");
  bad = good;
  const auto bpos = text.find("\"bytes\":", text.find("\"tensors\"")) + 8;
  auto& bdigit = bad[16 + bpos];
  bdigit = bdigit == '9' ? '8' : static_cast<std::uint8_t>(bdigit + 1);
  CHECK_THROWS(parse_efq(bad.data(), bad.size()), "header says");
  CHECK_THROWS(QuantFile::open(d.path / "missing.efq"), "cannot open");
}

// ---------------------------------------------------------------- mixed precision (overrides)

namespace {

std::size_t find_after(const std::vector<std::uint8_t>& f, const std::string& what, std::size_t from = 0) {
  const std::string text(f.begin(), f.end());
  return text.find(what, from);
}

// Golden prompt data in the tools/dump_reference.py layout, produced by the verified engine:
// the residual stream entering every layer (all positions if `full`, else the last one).
void write_engine_golden(const std::filesystem::path& model_dir, const std::filesystem::path& dir,
                         const std::vector<std::int32_t>& ids, bool full) {
  namespace fs = std::filesystem;
  const Model m = Model::load(model_dir);
  const ModelConfig& c = m.config();
  const auto D = static_cast<std::size_t>(c.hidden_size), L = static_cast<std::size_t>(c.num_hidden_layers);
  const auto V = static_cast<std::size_t>(c.vocab_size), S = ids.size();
  Transformer t(m, S);
  ForwardTrace tr;
  tr.capture = true;
  std::vector<float> embed_all(S * D), hidden_all(L * S * D), final_last(D), logits_last(V);
  for (std::size_t s = 0; s < S; ++s) {
    kern::embedding_bf16(embed_all.data() + s * D, m.embed_tokens(), static_cast<std::size_t>(ids[s]), D);
    const float* lg = t.forward(ids[s], s, true, &tr);
    for (std::size_t l = 0; l < L; ++l) {
      std::copy_n(tr.layer_out.data() + l * D, D, hidden_all.data() + (l * S + s) * D);
    }
    if (s + 1 == S) {
      final_last = tr.final_norm;
      logits_last.assign(lg, lg + V);
    }
  }
  std::vector<float> embed_last(embed_all.end() - static_cast<std::ptrdiff_t>(D), embed_all.end()), hidden_last(L * D);
  for (std::size_t l = 0; l < L; ++l) std::copy_n(hidden_all.data() + (l * S + S - 1) * D, D, hidden_last.data() + l * D);
  fs::create_directories(dir);
  std::string files;
  auto put = [&](const std::string& key, const void* data, std::size_t bytes, const std::string& dtype,
                 const std::string& shape) {
    std::ofstream(dir / (key + ".bin"), std::ios::binary).write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    files += (files.empty() ? "" : ",") + std::string("\"") + key + "\":{\"file\":\"" + key + ".bin\",\"dtype\":\"" +
             dtype + "\",\"shape\":" + shape + "}";
  };
  const std::string d = std::to_string(D), l = std::to_string(L), s = std::to_string(S);
  put("input_ids", ids.data(), S * 4, "i32", "[" + s + "]");
  put("embed_last", embed_last.data(), D * 4, "f32", "[" + d + "]");
  put("hidden_last", hidden_last.data(), L * D * 4, "f32", "[" + l + "," + d + "]");
  put("final_norm_last", final_last.data(), D * 4, "f32", "[" + d + "]");
  put("logits_last", logits_last.data(), V * 4, "f32", "[" + std::to_string(V) + "]");
  if (full) {
    put("embed_all", embed_all.data(), S * D * 4, "f32", "[" + s + "," + d + "]");
    put("hidden_all", hidden_all.data(), L * S * D * 4, "f32", "[" + l + "," + s + "," + d + "]");
  }
  std::ofstream(dir / "manifest.json") << "{\"format\":\"edgeforge-golden-v1\",\"name\":\"" << dir.filename().string()
                                       << "\",\"n_tokens\":" << S << ",\"files\":{" << files << "}}";
}

const GemvCheck* gemv_of(const VerifyQuantReport& r, const std::string& name) {
  for (const auto& g : r.gemv) {
    if (g.tensor == name) return &g;
  }
  return nullptr;
}

}  // namespace

TEST(quant_policy_globs_overrides_and_parsing) {
  CHECK(glob_match("model.layers.*.self_attn.v_proj.weight", "model.layers.12.self_attn.v_proj.weight"));
  CHECK(!glob_match("model.layers.*.self_attn.v_proj.weight", "model.layers.12.self_attn.k_proj.weight"));
  CHECK(glob_match("*", "") && glob_match("a*b*c", "aXbYc") && glob_match("abc", "abc"));
  CHECK(!glob_match("a*", "b") && !glob_match("abc", "abcd") && !glob_match("*.0.*", "model.layers.10.x"));
  CHECK(parse_overrides("none").empty() && parse_overrides("NONE").empty());
  const auto v = parse_overrides("model.layers.0.self_attn.k_proj.weight=q8_0,*.v_proj.weight=BF16");
  CHECK_EQ(v.size(), std::size_t{2});
  CHECK(v[0].type == QType::Q8_0 && v[1].type == QType::BF16 && v[1].pattern == "*.v_proj.weight");
  CHECK_EQ(overrides_to_arg(v), std::string("model.layers.0.self_attn.k_proj.weight=q8_0,*.v_proj.weight=bf16"));
  CHECK_EQ(overrides_to_arg({}), std::string("none"));
  CHECK_THROWS(parse_overrides("x"), "TENSOR=TYPE");
  CHECK_THROWS(parse_overrides("x="), "TENSOR=TYPE");
  CHECK_THROWS(parse_overrides("=q8_0"), "TENSOR=TYPE");
  CHECK_THROWS(parse_overrides("a=q8_0,"), "TENSOR=TYPE");
  CHECK_THROWS(parse_overrides("x=q3_k"), "type must be one of");

  QuantPolicy p;
  p.overrides = {{"model.layers.0.self_attn.v_proj.weight", QType::Q8_0}, {"model.layers.*.self_attn.v_proj.weight", QType::BF16}};
  auto c = policy_choice(p, "model.layers.0.self_attn.v_proj.weight", 64);
  CHECK(c.type == QType::BF16 && c.rule == "override");  // last match wins
  c = policy_choice(p, "model.layers.0.self_attn.k_proj.weight", 64);
  CHECK(c.type == QType::Q4_0 && c.rule == "layers");
  c = policy_choice(p, "model.layers.0.self_attn.v_proj.weight", 64, false);
  CHECK(c.type == QType::Q4_0 && c.rule == "layers");  // uniform baseline
  c = policy_choice(p, "model.layers.3.input_layernorm.weight", 64);
  CHECK(c.type == QType::F32 && c.rule == "norm");
  c = policy_choice(p, "lm_head.weight", 64);
  CHECK(c.type == QType::Q8_0 && c.rule == "lm_head");
  c = policy_choice(p, "model.embed_tokens.weight", 64);
  CHECK(c.type == QType::BF16 && c.rule == "embed");
  c = policy_choice(p, "model.layers.0.mlp.up_proj.weight", 30);
  CHECK(c.type == QType::BF16 && c.rule == "fallback-bf16");
  const std::vector<ExpectedTensor> ts = {{"model.layers.0.input_layernorm.weight", {32}},
                                          {"model.layers.0.self_attn.v_proj.weight", {16, 32}}};
  check_overrides(p, ts);
  QuantPolicy bad;
  bad.overrides = {{"model.layers.7.self_attn.v_proj.weight", QType::Q8_0}};
  CHECK_THROWS(check_overrides(bad, ts), "matches no tensor");
  bad.overrides = {{"*norm*", QType::Q8_0}};
  CHECK_THROWS(check_overrides(bad, ts), "always F32");
}

TEST(qfile_overrides_are_recorded_and_enforced) {
  const ef_test::Tiny t = tiny32();
  const ef_test::TinyDir d(t, 31);
  QuantizeOptions o;
  CHECK(o.overrides.empty());  // the library adds no model-specific exceptions by itself
  o.overrides = parse_overrides("model.layers.0.self_attn.k_proj.weight=q8_0,model.layers.0.self_attn.v_proj.weight=q8_0");
  const QuantizeResult r = quantize_model(d.path, d.path / "mixed.efq", o);
  // k/v are [16][32]: Q8_0 costs 16 x (34 - 18) = 256 bytes more per matrix than Q4_0
  CHECK_EQ(r.override_extra_bytes, std::int64_t{512});
  const QuantFile q = QuantFile::open(d.path / "mixed.efq");
  CHECK_EQ(q.policy().overrides.size(), std::size_t{2});
  CHECK_EQ(q.policy().name, std::string("custom"));
  CHECK(q.at("model.layers.0.self_attn.k_proj.weight").type == QType::Q8_0);
  CHECK(q.at("model.layers.0.self_attn.v_proj.weight").rule == "override");
  CHECK(q.at("model.layers.0.self_attn.q_proj.weight").type == QType::Q4_0);
  CHECK(q.at("model.layers.1.self_attn.k_proj.weight").type == QType::Q4_0);
  CHECK(q.at("model.layers.1.self_attn.k_proj.weight").rule == "layers");
  CHECK(q.at("lm_head.weight").rule == "lm_head" && q.at("model.norm.weight").rule == "norm");

  o.overrides.clear();  // uniform baseline: identical except the two overridden matrices
  quantize_model(d.path, d.path / "uniform.efq", o);
  const QuantFile u = QuantFile::open(d.path / "uniform.efq");
  CHECK(u.policy().overrides.empty() && u.policy().name == "uniform");
  CHECK(u.at("model.layers.0.self_attn.k_proj.weight").type == QType::Q4_0);
  CHECK_EQ(q.bytes_per_token() - u.bytes_per_token(), std::uint64_t{512});
  int same = 0;
  for (const auto& tq : q.tensors()) {
    const QTensor& tu = u.at(tq.name);
    if (tq.rule == "override") continue;
    same += tq.bytes == tu.bytes && std::memcmp(q.data(tq), u.data(tu), static_cast<std::size_t>(tq.bytes)) == 0;
  }
  CHECK_EQ(same, 21 - 2);

  // The reader re-derives every tensor's type from the policy in the header.
  const auto good = slurp(d.path / "mixed.efq");
  auto bad = good;
  std::size_t pos = find_after(bad, "\"overrides\":[{\"pattern\":\"model.layers.0.self_attn.k_proj.weight\",\"type\":\"");
  CHECK(pos != std::string::npos);
  pos += std::string("\"overrides\":[{\"pattern\":\"model.layers.0.self_attn.k_proj.weight\",\"type\":\"").size();
  bad[pos + 1] = '4';  // "Q8_0" -> "Q4_0": the policy no longer explains the stored Q8_0
  CHECK_THROWS(parse_efq(bad.data(), bad.size()), "policy gives Q4_0");
  bad = good;
  pos = find_after(bad, "\"overrides\":");
  bad[pos + 9] = 'z';  // a file without the overrides table (old pre-release build)
  CHECK_THROWS(parse_efq(bad.data(), bad.size()), "re-run edgeforge quantize");

  o.overrides = parse_overrides("model.layers.9.self_attn.v_proj.weight=q8_0");
  CHECK_THROWS(quantize_model(d.path, d.path / "x.efq", o), "matches no tensor");
  o.overrides = parse_overrides("model.norm.weight=q8_0");
  CHECK_THROWS(quantize_model(d.path, d.path / "x.efq", o), "always F32");
}

TEST(qfile_verify_measures_every_layer_on_real_activations) {
  const ef_test::Tiny t = tiny32();
  const ef_test::TinyDir d(t, 32);
  const auto golden = d.path / "golden";
  write_engine_golden(d.path, golden / "prompts" / "p0_full", {1, 5, 7, 3, 9, 2}, true);
  write_engine_golden(d.path, golden / "prompts" / "p1_last", {1, 4, 8, 6}, false);
  QuantizeOptions o;
  o.overrides = parse_overrides("model.layers.0.self_attn.k_proj.weight=q8_0,model.layers.0.self_attn.v_proj.weight=q8_0");
  quantize_model(d.path, d.path / "m.efq", o);
  VerifyQuantReport rep;
  VerifyQuantOptions vo;
  vo.qmodel = d.path / "m.efq";
  vo.model_dir = d.path;
  vo.golden_dir = golden;
  vo.threads = 3;
  vo.report = &rep;
  run_verify_quant(vo);
  // The activation bank re-derives each layer with the same kernels as the engine.
  CHECK(rep.activation_rel_err >= 0.0 && rep.activation_rel_err <= 1e-6);
  CHECK(rep.prompts_used == (std::vector<std::string>{"p0_full", "p1_last"}));
  CHECK_EQ(rep.gemv.size(), std::size_t{2 * 7});
  const GemvCheck* v0 = gemv_of(rep, "model.layers.1.self_attn.v_proj.weight");
  const GemvCheck* d1 = gemv_of(rep, "model.layers.1.mlp.down_proj.weight");
  CHECK(v0 != nullptr && v0->prompts == 2 && v0->positions == 7);  // q/k/v: all prompts
  CHECK(d1 != nullptr && d1->prompts == 1 && d1->positions == 6);  // o/gate/up/down: full prompt
  for (const auto& g : rep.gemv) CHECK(g.cosine > 0.9 && g.cosine <= 1.0 + 1e-12 && g.token_cosine <= g.cosine + 1e-12);
  CHECK_EQ(rep.overrides.size(), std::size_t{2});
  for (const auto& oc : rep.overrides) {
    CHECK(oc.stored == QType::Q8_0 && oc.baseline == QType::Q4_0 && oc.extra_bytes == 256);
    CHECK(oc.rmse_stored < oc.rmse_baseline && oc.cos_stored > oc.cos_baseline && oc.cos_baseline > 0.0);
  }
  // Same weights in the file as in the source: every output identical.
  o.layers = QType::BF16;
  o.overrides.clear();
  quantize_model(d.path, d.path / "bf16.efq", o);
  vo.qmodel = d.path / "bf16.efq";
  run_verify_quant(vo);
  for (const auto& g : rep.gemv) CHECK(g.cosine >= 1.0 - 1e-12 && g.type == QType::BF16);
  CHECK(rep.promotions.empty() && rep.suggested_override.empty());

  // Golden data of another model with the same shapes: caught, and no output error is reported.
  const ef_test::TinyDir other(t, 99);
  quantize_model(other.path, other.path / "o.efq", QuantizeOptions{});
  vo.qmodel = other.path / "o.efq";
  vo.model_dir = other.path;
  CHECK_EQ(run_verify_quant(vo), 1);
  CHECK(rep.activation_rel_err > 1e-3 && rep.gemv.empty());
  bool said = false;
  for (const auto& f : rep.failed_checks) said = said || f.find("do not belong to this model") != std::string::npos;
  CHECK(said);
  // ... and of a model with other shapes: a hard error, not a silent skip.
  const ef_test::TinyDir small(ef_test::Tiny{}, 5);
  quantize_model(small.path, small.path / "s.efq", QuantizeOptions{});
  vo.qmodel = small.path / "s.efq";
  vo.model_dir = small.path;
  CHECK_THROWS(run_verify_quant(vo), "different model");
}

TEST(qfile_verify_flags_a_sensitive_matrix_and_the_fix_is_exact) {
  // Layer-0 input channel 5 is huge (norm weight 200) while v_proj's column 5 is small (0.04):
  // Q4_0's per-block scale rounds that column to zero, so v_proj's output collapses even though
  // its weight error looks ordinary. This is the real TinyLlama failure mode in miniature.
  const ef_test::Tiny t = tiny32();
  const ef_test::TinyDir d(t, 33, false, [](const std::string& name, std::uint64_t i, float v) {
    if (name == "model.layers.0.input_layernorm.weight" && i == 5) return 200.0f;
    if (name == "model.layers.0.self_attn.v_proj.weight" && i % 32 == 5) return 0.04f;
    return v;
  });
  const auto golden = d.path / "golden";
  write_engine_golden(d.path, golden / "prompts" / "p0_full", {1, 5, 7, 3, 9, 2, 4, 8}, true);
  QuantizeOptions o;
  o.overrides.clear();
  quantize_model(d.path, d.path / "uniform.efq", o);
  VerifyQuantReport r1;
  VerifyQuantOptions vo;
  vo.qmodel = d.path / "uniform.efq";
  vo.model_dir = d.path;
  vo.golden_dir = golden;
  vo.report = &r1;
  CHECK_EQ(run_verify_quant(vo), 1);
  const std::string v0 = "model.layers.0.self_attn.v_proj.weight";
  CHECK(gemv_of(r1, v0) != nullptr && gemv_of(r1, v0)->cosine < 0.9);
  const Promotion* p = nullptr;
  for (const auto& x : r1.promotions) {
    if (x.tensor == v0) p = &x;
  }
  CHECK(p != nullptr);
  if (p == nullptr) return;
  CHECK(p->from == QType::Q4_0 && p->to == QType::Q8_0 && p->cos_from < 0.99 && p->cos_to >= 0.99);
  CHECK(p->reason.find("GEMV cosine") != std::string::npos);
  CHECK(r1.suggested_override.find(v0 + "=q8_0") != std::string::npos);

  // Apply the suggestion: the re-quantized file measures exactly what was predicted.
  o.overrides = parse_overrides(r1.suggested_override);
  quantize_model(d.path, d.path / "fixed.efq", o);
  VerifyQuantReport r2;
  vo.qmodel = d.path / "fixed.efq";
  vo.report = &r2;
  run_verify_quant(vo);
  for (const auto& x : r1.promotions) {
    const GemvCheck* g = gemv_of(r2, x.tensor);
    if (g != nullptr) CHECK(g->cosine == x.cos_to && g->type == x.to);
  }
  CHECK(r2.promotions.empty());
  for (const auto& g : r2.gemv) CHECK(g.cosine >= 0.99);
  bool listed = false;
  for (const auto& oc : r2.overrides) {
    if (oc.tensor == v0) listed = oc.cos_baseline < 0.99 && oc.cos_stored >= 0.99 && oc.stored == QType::Q8_0;
  }
  CHECK(listed);
}

TEST(quant_named_policy_tinyllama_is_explicit_and_guarded) {
  // The TinyLlama-1.1B config numbers (embedding untied); the tiny test model is something else.
  const ModelConfig tl = parse_model_config(json::parse(
      R"({"model_type":"llama","hidden_size":2048,"intermediate_size":5632,"num_hidden_layers":22,)"
      R"("num_attention_heads":32,"num_key_value_heads":4,"vocab_size":32000,"rms_norm_eps":1e-05,)"
      R"("rope_theta":10000.0,"max_position_embeddings":2048,"tie_word_embeddings":false})"));
  const ef_test::TinyDir d(tiny32(), 41);
  const ModelConfig other = load_model_config(d.path / "config.json");

  const PolicySelection a = select_policy("auto", tl);
  CHECK_EQ(a.name, std::string("tinyllama-1.1b-chat-v1.0"));
  CHECK_EQ(overrides_to_arg(a.overrides),
           std::string("model.layers.0.self_attn.k_proj.weight=q8_0,model.layers.0.self_attn.v_proj.weight=q8_0,"
                       "model.layers.1.self_attn.v_proj.weight=q8_0,model.layers.2.self_attn.v_proj.weight=q8_0,"
                       "model.layers.17.self_attn.v_proj.weight=q8_0,model.layers.18.self_attn.v_proj.weight=q8_0"));
  CHECK(a.note.find("auto") != std::string::npos);
  QuantPolicy p;
  p.overrides = a.overrides;
  check_overrides(p, expected_tensors(tl));  // every pattern names a real TinyLlama tensor
  std::size_t q8 = 0;
  for (const auto& e : expected_tensors(tl)) q8 += policy_choice(p, e.name, e.shape.back()).rule == "override";
  CHECK_EQ(q8, std::size_t{6});

  // Never applied to an unrelated architecture: auto falls back to uniform, explicit use is refused.
  const PolicySelection b = select_policy("auto", other);
  CHECK(b.name == "uniform" && b.overrides.empty());
  CHECK_THROWS(select_policy("tinyllama-1.1b-chat-v1.0", other), "different architecture");
  CHECK(select_policy("uniform", tl).overrides.empty());
  CHECK_THROWS(select_policy("q4-everything", tl), "unknown policy");
  ModelConfig tied = tl;
  tied.tie_word_embeddings = true;
  CHECK_EQ(select_policy("auto", tied).name, std::string("uniform"));

  // The policy name is recorded in the file.
  QuantizeOptions o;
  o.policy_name = "uniform";
  quantize_model(d.path, d.path / "u.efq", o);
  CHECK_EQ(QuantFile::open(d.path / "u.efq").policy().name, std::string("uniform"));
}
