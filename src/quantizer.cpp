// EdgeForge - Stage 5 model quantizer: model.safetensors -> .efq
#include "ef/quantizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <thread>

#include "ef/error.h"
#include "ef/format.h"
#include "ef/model_config.h"

namespace ef {

namespace {

std::string read_text(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw Error("cannot open '" + p.string() + "'");
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::uint64_t align_up(std::uint64_t v, std::uint64_t a) { return (v + a - 1) / a * a; }

// Fixed-width (12 chars) non-negative number, so the header can be rewritten in place.
std::string fixed_num(double v) {
  if (!(v >= 0.0) || !std::isfinite(v)) v = 0.0;
  if (v >= 9.999999e99) v = 9.999999e99;
  if (v != 0.0 && v < 1e-99) v = 0.0;
  return strf("%.6e", v);
}

template <class Fn>
void parallel_rows(std::uint64_t rows, unsigned threads, Fn fn) {
  const unsigned T = std::max(1u, std::min<unsigned>(threads, static_cast<unsigned>(std::max<std::uint64_t>(rows, 1))));
  if (T == 1) {
    fn(std::uint64_t{0}, rows);
    return;
  }
  std::vector<std::thread> pool;
  const std::uint64_t per = (rows + T - 1) / T;
  for (unsigned k = 0; k < T; ++k) {
    const std::uint64_t b = std::min<std::uint64_t>(rows, k * per), e = std::min<std::uint64_t>(rows, b + per);
    if (b < e) pool.emplace_back([&fn, b, e] { fn(b, e); });
  }
  for (auto& th : pool) th.join();
}

}  // namespace

// ------------------------------------------------------------------ named policies

bool NamedPolicy::matches(const ModelConfig& c) const noexcept {
  if (hidden == 0) return true;
  return c.hidden_size == hidden && c.intermediate_size == intermediate && c.num_hidden_layers == layers &&
         c.num_attention_heads == heads && c.num_key_value_heads == kv_heads && c.vocab_size == vocab &&
         !c.tie_word_embeddings;
}

const std::vector<NamedPolicy>& named_policies() {
  static const std::vector<NamedPolicy> v = [] {
    NamedPolicy uniform;
    uniform.name = "uniform";
    uniform.measured_on = "-";
    NamedPolicy tl;
    tl.name = "tinyllama-1.1b-chat-v1.0";
    // verify-quant on the real weights (2026-10-31): the uniform Q4_0 policy failed on exactly
    // these six matrices (layer-0 k_proj: weight rel RMSE 0.1256 > 0.12; the five v_proj:
    // GEMV cosine < 0.99 on real activations). With them in Q8_0: 12/12 checks, 154/154 matrices.
    tl.measured_on = "TinyLlama/TinyLlama-1.1B-Chat-v1.0 (model.safetensors sha256 6e6001da2106d475...)";
    tl.overrides = {{"model.layers.0.self_attn.k_proj.weight", QType::Q8_0},
                    {"model.layers.0.self_attn.v_proj.weight", QType::Q8_0},
                    {"model.layers.1.self_attn.v_proj.weight", QType::Q8_0},
                    {"model.layers.2.self_attn.v_proj.weight", QType::Q8_0},
                    {"model.layers.17.self_attn.v_proj.weight", QType::Q8_0},
                    {"model.layers.18.self_attn.v_proj.weight", QType::Q8_0}};
    tl.hidden = 2048;
    tl.intermediate = 5632;
    tl.layers = 22;
    tl.heads = 32;
    tl.kv_heads = 4;
    tl.vocab = 32000;
    return std::vector<NamedPolicy>{uniform, tl};
  }();
  return v;
}

PolicySelection select_policy(std::string_view policy, const ModelConfig& cfg) {
  const auto& all = named_policies();
  if (policy == "auto") {
    for (const auto& p : all) {
      if (p.hidden != 0 && p.matches(cfg)) {
        return {p.name, p.overrides,
                "auto: config matches its architecture; measured on " + p.measured_on +
                    " - other checkpoints: re-run verify-quant"};
      }
    }
    return {"uniform", {}, "auto: no named policy for this architecture"};
  }
  for (const auto& p : all) {
    if (p.name != policy) continue;
    if (!p.matches(cfg)) {
      throw Error("policy '" + p.name + "' was measured on " + p.measured_on +
                  "; this model's config has a different architecture (use --policy uniform)");
    }
    return {p.name, p.overrides, "explicit"};
  }
  std::string names;
  for (const auto& p : all) names += (names.empty() ? "" : ", ") + p.name;
  throw Error("unknown policy '" + std::string(policy) + "' (auto, " + names + ")");
}

QType choose_qtype(const std::string& name, std::uint64_t cols, const QuantizeOptions& opt) {
  return policy_choice(opt.policy(), name, cols).type;
}

void quantize_tensor_rows(const SafetensorsFile& st, const TensorInfo& t, QType type, bool search,
                          std::uint64_t row_begin, std::uint64_t row_end, std::uint8_t* out,
                          std::vector<double>* row_sse, std::vector<double>* row_ref, std::vector<double>* row_max) {
  const std::uint64_t cols = t.shape.empty() ? 1 : t.shape.back();
  const std::uint64_t row_bytes = qtype_row_bytes(type, cols);
  std::vector<float> src(static_cast<std::size_t>(cols)), back(static_cast<std::size_t>(cols));
  for (std::uint64_t r = row_begin; r < row_end; ++r) {
    st.read_f32(t, r * cols, cols, src.data());
    std::uint8_t* dst = out + (r - row_begin) * row_bytes;
    if (type == QType::BF16 && t.dtype == DType::BF16) {
      std::memcpy(dst, st.bytes(t).data() + r * cols * 2, static_cast<std::size_t>(cols * 2));  // exact copy
    } else {
      quantize_row(type, src.data(), static_cast<std::size_t>(cols), dst, search);
    }
    if (row_sse != nullptr) {
      dequantize_row(type, dst, static_cast<std::size_t>(cols), back.data());
      double sse = 0.0, ref = 0.0, mx = 0.0;
      for (std::uint64_t c = 0; c < cols; ++c) {
        const double e = static_cast<double>(src[c]) - static_cast<double>(back[c]);
        sse += e * e;
        ref += static_cast<double>(src[c]) * static_cast<double>(src[c]);
        mx = std::max(mx, std::fabs(e));
      }
      (*row_sse)[r] = sse;
      (*row_ref)[r] = ref;
      (*row_max)[r] = mx;
    }
  }
}

QuantizeResult quantize_model(const std::filesystem::path& model_dir, const std::filesystem::path& out_file,
                              const QuantizeOptions& opt,
                              const std::function<void(const QTensor&, std::size_t, std::size_t)>& progress) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::string config_text = read_text(model_dir / "config.json");
  const ModelConfig cfg = parse_model_config(json::parse(config_text));
  const SafetensorsFile st = SafetensorsFile::open(model_dir / "model.safetensors");
  const CheckReport rep = check_checkpoint(cfg, st.tensors());
  if (!rep.ok()) throw Error("checkpoint does not match config.json: " + rep.errors.front());
  const unsigned threads = opt.threads ? opt.threads : std::max(1u, std::thread::hardware_concurrency());

  QuantizeResult res;
  const QuantPolicy policy = opt.policy();
  const auto expected = expected_tensors(cfg);
  check_overrides(policy, expected);
  // ---- layout (forward-pass order)
  for (const auto& e : expected) {
    const TensorInfo& src = st.at(e.name);
    QTensor q;
    q.name = e.name;
    q.shape = e.shape;
    const PolicyChoice choice = policy_choice(policy, e.name, q.cols());
    q.type = choice.type;
    q.rule = choice.rule;
    if (choice.rule == "fallback-bf16") {
      res.notes.push_back(e.name + ": row length not a multiple of 32, kept as BF16");
    }
    q.bytes = qtype_row_bytes(q.type, q.cols()) * q.rows();
    if (choice.rule == "override") {
      const PolicyChoice base = policy_choice(policy, e.name, q.cols(), false);
      res.override_extra_bytes += static_cast<std::int64_t>(q.bytes) -
                                  static_cast<std::int64_t>(qtype_row_bytes(base.type, q.cols()) * q.rows());
    }
    res.source_tensor_bytes += src.nbytes();
    if (e.name != "model.embed_tokens.weight" || cfg.tie_word_embeddings) {
      res.bytes_per_token += q.bytes;
      res.source_bytes_per_token += src.nbytes();
    }
    res.tensors.push_back(std::move(q));
  }
  std::string trimmed = config_text;
  while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.back()))) trimmed.pop_back();
  std::string overrides_json = "[";
  for (const auto& o : policy.overrides) {
    overrides_json += (overrides_json.size() > 1 ? "," : "") + std::string("{\"pattern\":\"") +
                      json::escape(o.pattern) + "\",\"type\":\"" + qtype_name(o.type) + "\"}";
  }
  overrides_json += "]";
  auto build_header = [&](std::uint64_t data_start, bool with_stats) {
    std::string h = "{\"format\":\"edgeforge-quant\",\"version\":1,\n\"quant\":{\"layers\":\"" +
                    std::string(qtype_name(opt.layers)) + "\",\"lm_head\":\"" + qtype_name(opt.lm_head) +
                    "\",\"embed\":\"" + qtype_name(opt.embed) + "\",\"norms\":\"F32\",\"block\":32,\"scale\":\"" +
                    (opt.search ? "mse-search" : "absmax") + "\",\n\"policy\":\"" + json::escape(policy.name) +
                    "\",\"overrides\":" + overrides_json +
                    "},\n\"source\":{\"file\":\"model.safetensors\",\"bytes\":" + std::to_string(st.file_size()) +
                    "},\n\"config\":" + trimmed + ",\n\"tensors\":[\n";
    std::uint64_t off = data_start;
    for (std::size_t i = 0; i < res.tensors.size(); ++i) {
      const QTensor& q = res.tensors[i];
      off = align_up(off, kEfqAlign);
      h += "{\"name\":\"" + json::escape(q.name) + "\",\"type\":\"" + qtype_name(q.type) + "\",\"rule\":\"" +
           q.rule + "\",\"shape\":" + shape_to_string(q.shape) + ",\"offset\":" + std::to_string(off) +
           ",\"bytes\":" + std::to_string(q.bytes) +
           ",\"rel_rmse\":" + fixed_num(with_stats ? q.rel_rmse : 0.0) +
           ",\"max_abs_err\":" + fixed_num(with_stats ? q.max_abs_err : 0.0) + "}" +
           (i + 1 < res.tensors.size() ? ",\n" : "\n");
      off += q.bytes;
    }
    return h + "]}";
  };
  std::uint64_t data_start = 64;
  std::string header;
  for (int iter = 0; iter < 16; ++iter) {
    header = build_header(data_start, false);
    const std::uint64_t need = align_up(16 + header.size(), kEfqAlign);
    if (need <= data_start) break;
    data_start = need;
  }
  header.resize(static_cast<std::size_t>(data_start - 16), ' ');
  {
    std::uint64_t off = data_start;
    for (auto& q : res.tensors) {
      off = align_up(off, kEfqAlign);
      q.offset = off;
      off += q.bytes;
    }
    res.file_bytes = off;
  }

  // ---- write
  std::ofstream out(out_file, std::ios::binary | std::ios::trunc);
  if (!out) throw Error("cannot create '" + out_file.string() + "'");
  auto put_u32 = [&](std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.put(static_cast<char>((v >> (8 * i)) & 0xFF));
  };
  auto put_u64 = [&](std::uint64_t v) {
    for (int i = 0; i < 8; ++i) out.put(static_cast<char>((v >> (8 * i)) & 0xFF));
  };
  out.write(kEfqMagic, 4);
  put_u32(kEfqVersion);
  put_u64(data_start - 16);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));

  std::vector<std::uint8_t> buf;
  for (std::size_t i = 0; i < res.tensors.size(); ++i) {
    QTensor& q = res.tensors[i];
    const TensorInfo& src = st.at(q.name);
    const auto pos = static_cast<std::uint64_t>(out.tellp());
    if (pos > q.offset) throw Error("internal: layout overrun at '" + q.name + "'");
    static const char zeros[kEfqAlign] = {};
    out.write(zeros, static_cast<std::streamsize>(q.offset - pos));
    buf.assign(static_cast<std::size_t>(q.bytes), 0);
    const std::uint64_t rows = q.rows(), row_bytes = qtype_row_bytes(q.type, q.cols());
    std::vector<double> sse(rows), ref(rows), mx(rows);
    parallel_rows(rows, threads, [&](std::uint64_t b, std::uint64_t e) {
      quantize_tensor_rows(st, src, q.type, opt.search, b, e, buf.data() + b * row_bytes, &sse, &ref, &mx);
    });
    QuantError err;
    for (std::uint64_t r = 0; r < rows; ++r) {  // fixed order: stats are reproducible
      err.sse += sse[r];
      err.ref_ss += ref[r];
      err.max_abs_err = std::max(err.max_abs_err, mx[r]);
    }
    q.rel_rmse = err.rel_rmse();
    q.max_abs_err = err.max_abs_err;
    out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    if (!out) throw Error("write failed for '" + out_file.string() + "' (disk full?)");
    if (progress) progress(q, i, res.tensors.size());
  }
  // Final header with the measured errors (same byte length as the placeholder).
  std::string final_header = build_header(data_start, true);
  final_header.resize(static_cast<std::size_t>(data_start - 16), ' ');
  if (final_header.size() != header.size()) throw Error("internal: header size changed");
  out.seekp(16);
  out.write(final_header.data(), static_cast<std::streamsize>(final_header.size()));
  out.close();
  if (!out) throw Error("could not finish writing '" + out_file.string() + "'");
  res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return res;
}

}  // namespace ef
