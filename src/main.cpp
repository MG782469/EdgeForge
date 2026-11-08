// EdgeForge - command-line entry point (Stage 1: inspect, verify, membench, sysinfo).
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "ef/eos_study.h"
#include "ef/error.h"
#include "ef/format.h"
#include "ef/json.h"
#include "ef/membench.h"
#include "ef/model_config.h"
#include "ef/safetensors.h"
#include "ef/sysinfo.h"
#include "ef/verify_math.h"
#include "ef/golden.h"
#include "ef/model.h"
#include "ef/verify_forward.h"
#include "ef/chat.h"
#include "ef/eval_quant.h"
#include "ef/bench_decode.h"
#include "ef/qkernels.h"
#include "ef/threadpool.h"
#include "ef/tokenizer.h"
#include "ef/verify_chat.h"
#include "ef/verify_tokenizer.h"
#include "ef/quantizer.h"
#include "ef/verify_quant.h"
#include <iostream>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;
using namespace ef;

namespace {

constexpr const char* kVersion = "0.8.0 (stage 8)";

// Acceptance numbers for TinyLlama-1.1B (computed from its config; see docs).
struct KnownModel {
  const char* name;
  std::int64_t hidden, intermediate, layers, heads, kv_heads, vocab;
  std::uint64_t tensors, params, bytes_bf16;
};
constexpr KnownModel kTinyLlama = {"TinyLlama-1.1B", 2048, 5632, 22, 32, 4, 32000, 201, 1'100'048'384ull,
                                   2'200'096'768ull};

void print_usage() {
  std::printf(
      "EdgeForge %s - CPU inference engine for TinyLlama\n\n"
      "Usage:\n"
      "  edgeforge inspect  --model-dir DIR [--list] [--tensor NAME] [--count N]\n"
      "  edgeforge verify   --model-dir DIR [--golden DIR] [--quick]\n"
      "  edgeforge membench [--mb 1024] [--threads 1,2,4,8 | --sweep] [--runs 5] [--passes 3]\n"
      "                     [--warmup 1] [--force] [--json FILE]\n"
      "  edgeforge verify-math --golden DIR [--model-dir DIR] [--prompt p0_factual] [--bench FILE]\n"
      "  edgeforge verify-forward --golden DIR --model-dir DIR [--prompt NAME]\n"
      "  edgeforge generate (--model-dir DIR | --qmodel FILE.efq) (--ids 1,529,... | --ids-from GOLDEN_PROMPT_DIR)\n"
      "                     [--max-new 32] [--eos ID] [--bench FILE] [--threads N] [--kernels auto|avx2|scalar]\n"
      "  edgeforge chat     (--model-dir DIR | --qmodel FILE.efq --tokenizer-dir DIR) [--prompt TEXT]\n"
      "                     [--system TEXT | --no-system] [--max-new 256] [--ctx 2048] [--threads N]\n"
      "                     [--kernels auto|avx2|scalar] [--temp 0] [--top-k 0] [--top-p 1] [--rep-penalty 1] [--seed 0]\n"
      "                     [--no-stop-markers] [--stop TEXT] [--repeat-count 3] [--repeat-span 12] [--eos-min-p 0]\n"
      "  edgeforge verify-chat --golden DIR (--model-dir DIR | --qmodel FILE.efq --tokenizer-dir DIR) [--threads N]\n"
      "  edgeforge eos-study --golden tests/golden/chat_eos_study.json --model-dir DIR [--qmodel FILE.efq]\n"
      "                     [--labels LABELS.json] [--out bench/eos_study] [--threads N]\n"
      "  edgeforge eos-study --results bench/eos_study_results.json --labels LABELS.json [--out bench/eos_study]\n"
      "  edgeforge tokenize --tokenizer-dir DIR (--text TEXT | --text-file FILE [--ids-out IDS.txt]) [--no-special]\n"
      "  edgeforge verify-tokenizer --tokenizer-dir DIR --golden DIR [--model-dir DIR]\n"
      "  edgeforge quantize --model-dir DIR --out FILE.efq [--policy auto|uniform|NAME] [--override TENSOR=TYPE,...]\n"
      "                     [--layers q4_0] [--lm-head q8_0] [--embed bf16] [--no-search] [--threads N]\n"
      "  edgeforge verify-quant --qmodel FILE.efq --model-dir DIR [--golden DIR] [--threads N]\n"
      "  edgeforge quant-analyze --model-dir DIR [--golden DIR]\n"
      "  edgeforge eval-quant --qmodel FILE.efq [--model-dir DIR] [--golden DIR] [--text FILE] [--max-tokens 128]\n"
      "                     [--window 512 --chunks 8] [--tokenizer-dir DIR] [--threads N] [--kernels auto|avx2|scalar]\n"
      "  edgeforge bench-decode (--qmodel FILE.efq | --model-dir DIR) [--ids-from DIR | --ids 1,...] [--tokens 32]\n"
      "                     [--threads 2,4,6,12] [--no-ladder] [--bench bench/membench.json]\n"
      "  edgeforge profile  (--qmodel FILE.efq | --model-dir DIR) [--ids-from DIR | --ids 1,...] [--tokens 32]\n"
      "                     [--threads N] [--kernels auto|avx2|scalar]\n"
      "  edgeforge sysinfo\n\n"
      "inspect   Print config, header and tensor table; --tensor shows values + stats of one tensor.\n"
      "verify    Check the checkpoint against the config and the TinyLlama acceptance numbers;\n"
      "          scans every weight for NaN/Inf (skip with --quick); with --golden, compares\n"
      "          weight_probe.json from tools/dump_reference.py bit-for-bit. Exit code 0 = PASS.\n"
      "membench  Multithreaded memory read bandwidth (default: 1 GiB, thread sweep).\n"
      "verify-math  Stage 2: every math kernel (RoPE, softmax/attention, SwiGLU, residual; with\n"
      "          --model-dir also embedding, RMSNorm, all layer-0 GEMVs, a chained layer 0, final\n"
      "          norm and lm_head) against PyTorch golden outputs. Exit code 0 = PASS.\n"
      "verify-forward  Stage 3: all layers + KV-cache decoding vs golden (hidden states per layer,\n"
      "          logits at every position, decode logits, greedy tokens). Exit code 0 = PASS.\n"
      "generate  Greedy decoding from token ids with tok/s and GB/s.\n"
      "chat      Interactive chat (TinyLlama chat template, streaming, KV cache reused across turns);\n"
      "          --prompt answers one message and exits. Greedy by default (--temp > 0 to sample).\n"
      "          A reply ends at EOS, or when the model writes a role marker (<|user|> ...; the reply is\n"
      "          cut before it), or repeats a --repeat-span token span --repeat-count times (0 = off);\n"
      "          --max-new is the last bound. Default system prompt: a short, direct reply.\n"
      "verify-chat  Stage 8: chat replies vs Hugging Face greedy generation (tools/hf_chat_reference.py):\n"
      "          prompt ids identical, generated ids identical up to near-ties; then the default chat\n"
      "          settings on the same cases, with how and where each reply ended. Exit code 0 = PASS.\n"
      "eos-study  Stage 8: does an --eos-min-p threshold end replies better without cutting real answers?\n"
      "          Replays the Hugging Face traces (BF16) and the 4-bit engine's own traces at every threshold,\n"
      "          compiles code replies, writes <out>.md, <out>_results.json and a labels file for the human\n"
      "          review; with --labels applies the predeclared decision rule (--results: no model runs).\n"
      "verify-tokenizer  Stage 4: tokenizer, chat template and decoder vs Hugging Face golden data;\n"
      "          with --model-dir also prompt text -> EdgeForge -> reply text vs PyTorch.\n"
      "quantize  Stage 5: BF16 checkpoint -> .efq (Q4_0 layers, Q8_0 lm_head, BF16 embedding, F32 norms).\n"
      "          --policy auto (default) picks a named, measured policy only if the config matches its\n"
      "          architecture (tinyllama-1.1b-chat-v1.0: 6 matrices in Q8_0), else uniform. --override adds\n"
      "          per-tensor exceptions on top (exact names or '*' globs, last match wins).\n"
      "verify-quant  Reconstruction error of every tensor, lossless parts bit-exact, determinism,\n"
      "          file size and bytes/token; with --golden also the output error of every layer\n"
      "          matrix on real activations, overrides vs uniform, and a measured fix for failures.\n"
      "quant-analyze  Error of Q4 g32/g64/g128, absmax vs MSE scale search, and Q8 on sampled weights.\n"
      "eval-quant  Stage 6: the engine on the packed .efq weights - kernel check on the real weights,\n"
      "          KL / top-1 vs PyTorch (golden, teacher-forced), greedy continuations, text perplexity\n"
      "          vs the BF16 engine (--model-dir), tok/s, GB/s and peak RSS. Exit code 0 = PASS.\n"
      "          --window W --chunks N scores the text like llama.cpp's llama-perplexity -c W --chunks N.\n"
      "--qmodel  generate / chat run the quantized engine straight from an .efq file.\n"
      "bench-decode  Stage 7 speed-up ladder: scalar 1 thread -> AVX2 1 thread -> AVX2 N threads;\n"
      "          tok/s, GB/s, %% of the membench baseline; checks every thread count gives identical logits.\n"
      "          --no-ladder: only the SIMD backend at the --threads counts (quick comparisons).\n"
      "profile   Stage 8: per-op time of each decoded token (matmuls, attention, norms, lm_head) and\n"
      "          the GB/s each group of matrices streams.\n"
      "--threads N   engine threads (default: physical cores). --kernels: auto = AVX2 if built in.\n",
      kVersion);
}

// ------------------------------------------------------------------ args

struct Args {
  std::string command;
  std::map<std::string, std::string> opts;
  std::set<std::string> flags;

  bool has(const std::string& k) const { return opts.count(k) || flags.count(k); }
  std::string get(const std::string& k, const std::string& def = "") const {
    auto it = opts.find(k);
    return it == opts.end() ? def : it->second;
  }
  double get_double(const std::string& k, double def) const {
    auto it = opts.find(k);
    if (it == opts.end()) return def;
    char* end = nullptr;
    const double v = std::strtod(it->second.c_str(), &end);
    if (end == it->second.c_str() || *end != '\0') throw Error("--" + k + " expects a number, got '" + it->second + "'");
    return v;
  }
  long long get_int(const std::string& k, long long def) const {
    auto it = opts.find(k);
    if (it == opts.end()) return def;
    char* end = nullptr;
    const long long v = std::strtoll(it->second.c_str(), &end, 10);
    if (end == it->second.c_str() || *end != '\0') throw Error("--" + k + " expects an integer, got '" + it->second + "'");
    return v;
  }
};

Args parse_args(int argc, char** argv) {
  static const std::map<std::string, std::set<std::string>> value_opts = {
      {"inspect", {"model-dir", "tensor", "count"}},
      {"verify", {"model-dir", "golden"}},
      {"membench", {"mb", "threads", "runs", "passes", "warmup", "json"}},
      {"sysinfo", {}},
      {"verify-math", {"golden", "model-dir", "prompt", "bench"}},
      {"verify-forward", {"golden", "model-dir", "prompt", "kernels"}},
      {"generate", {"model-dir", "qmodel", "ids", "ids-from", "max-new", "eos", "bench", "threads", "kernels"}},
      {"chat", {"model-dir", "qmodel", "tokenizer-dir", "threads", "kernels", "prompt", "system", "max-new", "ctx", "temp", "top-k", "top-p",
                "rep-penalty", "seed", "stop", "repeat-count", "repeat-span", "eos-min-p"}},
      {"verify-chat", {"golden", "model-dir", "qmodel", "tokenizer-dir", "threads", "kernels"}},
      {"eos-study", {"golden", "model-dir", "qmodel", "tokenizer-dir", "results", "labels", "out", "threads", "kernels"}},
      {"tokenize", {"tokenizer-dir", "text", "text-file", "ids-out"}},
      {"verify-tokenizer", {"tokenizer-dir", "golden", "model-dir"}},
      {"quantize", {"model-dir", "out", "policy", "layers", "lm-head", "embed", "override", "threads"}},
      {"verify-quant", {"qmodel", "model-dir", "golden", "threads"}},
      {"quant-analyze", {"model-dir", "golden"}},
      {"eval-quant", {"qmodel", "model-dir", "golden", "text", "max-tokens", "window", "chunks", "tokenizer-dir", "threads",
                      "kernels"}},
      {"bench-decode", {"qmodel", "model-dir", "ids", "ids-from", "tokens", "threads", "bench"}},
      {"profile", {"qmodel", "model-dir", "ids", "ids-from", "tokens", "threads", "kernels", "bench"}},
  };
  static const std::map<std::string, std::set<std::string>> flag_opts = {
      {"inspect", {"list"}}, {"verify", {"quick"}}, {"membench", {"sweep", "force"}}, {"sysinfo", {}},
      {"verify-math", {}}, {"verify-forward", {}}, {"generate", {}}, {"chat", {"no-system", "no-stop-markers"}}, {"verify-chat", {}}, {"eos-study", {}},
      {"tokenize", {"no-special"}},
      {"verify-tokenizer", {}}, {"quantize", {"no-search"}}, {"verify-quant", {}}, {"quant-analyze", {}},
      {"eval-quant", {}}, {"bench-decode", {"no-ladder"}}, {"profile", {}},
  };
  Args a;
  if (argc < 2) return a;
  std::string cmd = argv[1];
  if (cmd.rfind("--", 0) == 0 && value_opts.count(cmd.substr(2))) cmd = cmd.substr(2);  // --inspect form
  a.command = cmd;
  if (cmd == "help" || cmd == "--help" || cmd == "-h" || cmd == "--version" || cmd == "version") return a;
  if (!value_opts.count(cmd)) throw Error("unknown command '" + std::string(argv[1]) + "' (try: edgeforge --help)");
  const auto& vals = value_opts.at(cmd);
  const auto& flags = flag_opts.at(cmd);
  for (int i = 2; i < argc; ++i) {
    std::string s = argv[i];
    if (s.rfind("--", 0) != 0) throw Error("unexpected argument '" + s + "'");
    s = s.substr(2);
    std::string value;
    const auto eq = s.find('=');
    const bool inline_value = eq != std::string::npos;
    if (inline_value) {
      value = s.substr(eq + 1);
      s = s.substr(0, eq);
    }
    if (flags.count(s)) {
      if (inline_value) throw Error("--" + s + " does not take a value");
      a.flags.insert(s);
    } else if (vals.count(s)) {
      if (!inline_value) {
        if (i + 1 >= argc) throw Error("--" + s + " needs a value");
        value = argv[++i];
      }
      a.opts[s] = value;
    } else {
      throw Error("unknown option --" + s + " for '" + cmd + "'");
    }
  }
  return a;
}

// ------------------------------------------------------------------ model files

struct ModelFiles {
  fs::path dir, config, weights;
};

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

ModelFiles resolve_model_dir(const Args& a) {
  if (!a.has("model-dir")) throw Error("--model-dir DIR is required (folder with config.json + model.safetensors)");
  ModelFiles m;
  m.dir = fs::path(a.get("model-dir"));
  if (!fs::is_directory(m.dir)) throw Error("model directory '" + m.dir.string() + "' does not exist");
  m.config = m.dir / "config.json";
  m.weights = m.dir / "model.safetensors";
  if (!fs::exists(m.config)) {
    throw Error("config.json not found in '" + m.dir.string() +
                "'. Download with: hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir <dir>");
  }
  if (!fs::exists(m.weights)) {
    if (fs::exists(m.dir / "model.safetensors.index.json")) {
      throw Error("sharded checkpoint (model.safetensors.index.json) is not supported yet");
    }
    if (fs::exists(m.dir / "pytorch_model.bin")) {
      throw Error("only pytorch_model.bin found; EdgeForge reads model.safetensors");
    }
    throw Error("model.safetensors not found in '" + m.dir.string() + "'");
  }
  if (lower(fs::absolute(m.dir).string()).find("onedrive") != std::string::npos) {
    std::fprintf(stderr,
                 "warning: the model folder is inside OneDrive. Cloud-only files are downloaded on first\n"
                 "         read and syncing slows I/O; keep models/ outside OneDrive for benchmarks.\n");
  }
  return m;
}

// ------------------------------------------------------------------ stats

struct TensorStats {
  double min = std::numeric_limits<double>::infinity(), max = -std::numeric_limits<double>::infinity(), mean = 0.0, m2 = 0.0, absmax = 0.0;
  double sum = 0.0, sum_c = 0.0, sum_abs = 0.0, sum_abs_c = 0.0;  // Neumaier sums
  std::uint64_t n = 0, nan = 0, inf = 0, zeros = 0;

  static void neumaier(double& s, double& c, double x) {
    const double t = s + x;
    c += (std::fabs(s) >= std::fabs(x)) ? (s - t) + x : (x - t) + s;
    s = t;
  }
  void add(float fv) {
    const double v = static_cast<double>(fv);
    if (std::isnan(v)) { ++nan; return; }
    if (std::isinf(v)) { ++inf; return; }
    if (v == 0.0) ++zeros;
    ++n;
    min = std::min(min, v);
    max = std::max(max, v);
    absmax = std::max(absmax, std::fabs(v));
    const double d = v - mean;  // Welford
    mean += d / static_cast<double>(n);
    m2 += d * (v - mean);
    neumaier(sum, sum_c, v);
    neumaier(sum_abs, sum_abs_c, std::fabs(v));
  }
  double total() const { return sum + sum_c; }
  double total_abs() const { return sum_abs + sum_abs_c; }
  double stddev() const { return n > 1 ? std::sqrt(m2 / static_cast<double>(n)) : 0.0; }
};

TensorStats tensor_stats(const SafetensorsFile& st, const TensorInfo& t) {
  TensorStats s;
  constexpr std::uint64_t kChunk = 1 << 16;
  std::vector<float> buf(kChunk);
  for (std::uint64_t i = 0; i < t.numel; i += kChunk) {
    const std::uint64_t n = std::min(kChunk, t.numel - i);
    st.read_f32(t, i, n, buf.data());
    for (std::uint64_t k = 0; k < n; ++k) s.add(buf[k]);
  }
  return s;
}

// ------------------------------------------------------------------ inspect

void print_config(const ModelConfig& c) {
  std::string arch = c.architectures.empty() ? "?" : c.architectures.front();
  std::printf("config.json      : %s (%s), torch_dtype %s\n", c.model_type.c_str(), arch.c_str(),
              c.torch_dtype.empty() ? "?" : c.torch_dtype.c_str());
  std::printf("  hidden %lld | intermediate %lld | layers %lld | heads %lld | kv heads %lld | head_dim %lld\n",
              static_cast<long long>(c.hidden_size), static_cast<long long>(c.intermediate_size),
              static_cast<long long>(c.num_hidden_layers), static_cast<long long>(c.num_attention_heads),
              static_cast<long long>(c.num_key_value_heads), static_cast<long long>(c.head_dim));
  std::printf("  vocab %lld | max positions %lld | rms_norm_eps %g | rope_theta %g | tied embeddings: %s\n",
              static_cast<long long>(c.vocab_size), static_cast<long long>(c.max_position_embeddings),
              c.rms_norm_eps, c.rope_theta, c.tie_word_embeddings ? "yes" : "no");
  std::printf("  derived: kv_dim %lld | GQA: %lld query heads share each KV head | bos %lld eos %lld\n",
              static_cast<long long>(c.kv_dim()), static_cast<long long>(c.gqa_group()),
              static_cast<long long>(c.bos_token_id), static_cast<long long>(c.eos_token_id));
}

void print_header_summary(const SafetensorsFile& st) {
  const auto& h = st.header();
  std::uint64_t params = 0, bytes = 0;
  std::map<std::string, std::uint64_t> dtypes;
  for (const auto& t : h.tensors) {
    params += t.numel;
    bytes += t.nbytes();
    ++dtypes[dtype_name(t.dtype)];
  }
  std::printf("model.safetensors: %s bytes | header %s bytes | data region starts at byte %s\n",
              with_commas(st.file_size()).c_str(), with_commas(h.header_len).c_str(),
              with_commas(h.data_offset).c_str());
  std::printf("  tensors %zu | parameters %s | tensor data %s bytes (%s)\n", h.tensors.size(),
              with_commas(params).c_str(), with_commas(bytes).c_str(), human_bytes(bytes).c_str());
  std::string dt;
  for (const auto& [k, v] : dtypes) dt += (dt.empty() ? "" : ", ") + k + " x" + std::to_string(v);
  std::printf("  dtypes: %s\n", dt.c_str());
  if (!h.metadata.empty()) {
    std::string md;
    for (const auto& [k, v] : h.metadata) md += (md.empty() ? "" : ", ") + k + "=" + v;
    std::printf("  metadata: %s\n", md.c_str());
  }
}

void print_tensor_row(const SafetensorsFile& st, const TensorInfo& t) {
  std::printf("  %-52s %-5s %-14s off %13s  %11s B\n", t.name.c_str(), dtype_name(t.dtype),
              shape_to_string(t.shape).c_str(), with_commas(st.header().data_offset + t.begin).c_str(),
              with_commas(t.nbytes()).c_str());
}

int cmd_inspect(const Args& a) {
  const ModelFiles m = resolve_model_dir(a);
  std::printf("EdgeForge %s - inspect\nmodel dir        : %s\n", kVersion, m.dir.string().c_str());
  const ModelConfig cfg = load_model_config(m.config);
  print_config(cfg);
  const auto t0 = std::chrono::steady_clock::now();
  const SafetensorsFile st = SafetensorsFile::open(m.weights);
  const double open_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  print_header_summary(st);
  std::printf("  header parsed + validated in %.2f ms (weights are memory-mapped, not loaded)\n", open_ms);

  const CheckReport rep = check_checkpoint(cfg, st.tensors());
  std::printf("checks vs config : %s (%llu/%zu expected tensors, %s params)\n", rep.ok() ? "PASS" : "FAIL",
              static_cast<unsigned long long>(rep.matched_tensors), expected_tensors(cfg).size(),
              with_commas(rep.matched_params).c_str());
  for (const auto& e : rep.errors) std::printf("  error: %s\n", e.c_str());
  for (const auto& w : rep.warnings) std::printf("  warning: %s\n", w.c_str());

  if (a.has("tensor")) {
    const TensorInfo& t = st.at(a.get("tensor"));
    const long long count = std::max(0LL, a.get_int("count", 8));
    std::printf("\ntensor %s\n", t.name.c_str());
    std::printf("  dtype %s | shape %s | %s elements | %s bytes\n", dtype_name(t.dtype),
                shape_to_string(t.shape).c_str(), with_commas(t.numel).c_str(), with_commas(t.nbytes()).c_str());
    std::printf("  data_offsets [%s, %s) -> file bytes [%s, %s)\n", with_commas(t.begin).c_str(),
                with_commas(t.end).c_str(), with_commas(st.header().data_offset + t.begin).c_str(),
                with_commas(st.header().data_offset + t.end).c_str());
    if (dtype_converts_to_f32(t.dtype)) {
      const std::uint64_t n = std::min<std::uint64_t>(static_cast<std::uint64_t>(count), t.numel);
      std::printf("  first %llu values (raw bits -> exact f32):\n", static_cast<unsigned long long>(n));
      for (std::uint64_t i = 0; i < n; ++i) {
        const unsigned bits = st.raw_bits(t, i);
        std::printf("    [%llu] 0x%0*x -> %.9g\n", static_cast<unsigned long long>(i),
                    static_cast<int>(dtype_size(t.dtype) * 2), bits, static_cast<double>(st.value_f32(t, i)));
      }
      const auto s0 = std::chrono::steady_clock::now();
      const TensorStats s = tensor_stats(st, t);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
      std::printf("  stats: min %.6g | max %.6g | mean %.6g | std %.6g | absmax %.6g\n", s.min, s.max, s.mean,
                  s.stddev(), s.absmax);
      std::printf("         sum %.17g | sum|x| %.17g | zeros %llu | NaN %llu | Inf %llu\n", s.total(), s.total_abs(),
                  static_cast<unsigned long long>(s.zeros), static_cast<unsigned long long>(s.nan),
                  static_cast<unsigned long long>(s.inf));
      std::printf("  scanned in %.1f ms; process peak RSS %s (whole model is %s)\n", ms,
                  human_bytes(peak_rss_bytes()).c_str(), human_bytes(st.file_size()).c_str());
    }
    return rep.ok() ? 0 : 1;
  }

  std::printf("\n%s:\n", a.has("list") ? "all tensors" : "tensors (layer 0 + non-layer; --list for all)");
  for (const auto& t : st.tensors()) {
    const bool layer = t.name.rfind("model.layers.", 0) == 0;
    const bool layer0 = t.name.rfind("model.layers.0.", 0) == 0;
    if (a.has("list") || !layer || layer0) print_tensor_row(st, t);
  }
  return rep.ok() ? 0 : 1;
}

// ------------------------------------------------------------------ verify

struct Checker {
  int pass = 0, fail = 0, skip = 0;
  void report(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (ok) {
      ++pass;
    } else {
      ++fail;
    }
  }
  void skipped(const std::string& what) {
    std::printf("  [SKIP] %s\n", what.c_str());
    ++skip;
  }
};

void verify_weight_probe(const SafetensorsFile& st, const fs::path& probe_path, Checker& chk) {
  std::ifstream in(probe_path, std::ios::binary);
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const json::Value root = json::parse(text);
  const auto& tensors = root.at("tensors");
  std::uint64_t samples = 0, bit_exact = 0;
  int sums_ok = 0, n_tensors = 0;
  std::vector<std::string> problems;
  for (const auto& e : tensors.items()) {
    ++n_tensors;
    const std::string name = e.at("name").as_string();
    const TensorInfo* t = st.find(name);
    if (t == nullptr) {
      problems.push_back("probe tensor '" + name + "' missing in checkpoint");
      continue;
    }
    if (e.at("dtype").as_string() != dtype_name(t->dtype)) problems.push_back(name + ": dtype differs");
    std::vector<std::uint64_t> shape;
    for (const auto& d : e.at("shape").items()) shape.push_back(d.as_uint64());
    if (shape != t->shape) problems.push_back(name + ": shape differs");
    for (const auto& smp : e.at("samples").items()) {
      const std::uint64_t idx = smp[0].as_uint64();
      const std::uint64_t bits = smp[1].as_uint64();
      ++samples;
      if (idx < t->numel && st.raw_bits(*t, idx) == bits) {
        ++bit_exact;
      } else if (problems.size() < 12) {
        problems.push_back(strf("%s[%llu]: python 0x%llx vs C++ 0x%x", name.c_str(), static_cast<unsigned long long>(idx),
                                static_cast<unsigned long long>(bits), idx < t->numel ? st.raw_bits(*t, idx) : 0u));
      }
    }
    const TensorStats s = tensor_stats(st, *t);
    const double py_sum = e.at("sum_f64").as_double();
    const double py_abs = e.at("sum_abs_f64").as_double();
    const double tol = 1e-9 * std::max(1.0, py_abs);
    if (std::fabs(s.total() - py_sum) <= tol && std::fabs(s.total_abs() - py_abs) <= tol) {
      ++sums_ok;
    } else {
      problems.push_back(strf("%s: sum C++ %.17g vs python %.17g", name.c_str(), s.total(), py_sum));
    }
  }
  chk.report(samples > 0 && bit_exact == samples,
             strf("golden weight probe: %llu/%llu sampled values bit-exact vs Python safetensors",
                  static_cast<unsigned long long>(bit_exact), static_cast<unsigned long long>(samples)));
  chk.report(sums_ok == n_tensors && n_tensors > 0,
             strf("golden weight probe: full-tensor sums match for %d/%d tensors", sums_ok, n_tensors));
  for (const auto& p : problems) std::printf("         %s\n", p.c_str());
}

int cmd_verify(const Args& a) {
  const ModelFiles m = resolve_model_dir(a);
  std::printf("EdgeForge %s - verify\nmodel dir: %s\n", kVersion, m.dir.string().c_str());
  Checker chk;

  ModelConfig cfg;
  try {
    cfg = load_model_config(m.config);
    chk.report(true, "config.json parses and is a supported Llama config");
  } catch (const Error& e) {
    chk.report(false, std::string("config.json: ") + e.what());
    std::printf("RESULT: FAIL\n");
    return 1;
  }
  SafetensorsFile st;
  try {
    st = SafetensorsFile::open(m.weights);
    chk.report(true, strf("safetensors header valid: %zu tensors exactly tile the %s-byte data region",
                          st.tensors().size(), with_commas(st.header().data_size).c_str()));
  } catch (const Error& e) {
    chk.report(false, e.what());
    std::printf("RESULT: FAIL\n");
    return 1;
  }

  const CheckReport rep = check_checkpoint(cfg, st.tensors());
  chk.report(rep.ok(), strf("tensor names + shapes match config (%llu/%zu)",
                            static_cast<unsigned long long>(rep.matched_tensors), expected_tensors(cfg).size()));
  for (const auto& e : rep.errors) std::printf("         %s\n", e.c_str());
  for (const auto& w : rep.warnings) std::printf("         warning: %s\n", w.c_str());

  std::uint64_t params = 0, bytes = 0;
  bool all_bf16 = true;
  for (const auto& t : st.tensors()) {
    params += t.numel;
    bytes += t.nbytes();
    all_bf16 = all_bf16 && t.dtype == DType::BF16;
  }
  chk.report(params == expected_param_count(cfg),
             strf("parameter count %s matches config (%s)", with_commas(params).c_str(),
                  with_commas(expected_param_count(cfg)).c_str()));

  const bool is_tiny = cfg.hidden_size == kTinyLlama.hidden && cfg.intermediate_size == kTinyLlama.intermediate &&
                       cfg.num_hidden_layers == kTinyLlama.layers && cfg.num_attention_heads == kTinyLlama.heads &&
                       cfg.num_key_value_heads == kTinyLlama.kv_heads && cfg.vocab_size == kTinyLlama.vocab;
  if (is_tiny) {
    chk.report(st.tensors().size() == kTinyLlama.tensors,
               strf("tensor count %zu == %llu (TinyLlama acceptance)", st.tensors().size(),
                    static_cast<unsigned long long>(kTinyLlama.tensors)));
    chk.report(params == kTinyLlama.params, strf("parameters %s == %s (TinyLlama acceptance)",
                                                 with_commas(params).c_str(), with_commas(kTinyLlama.params).c_str()));
    chk.report(bytes == kTinyLlama.bytes_bf16, strf("tensor bytes %s == %s (TinyLlama acceptance)",
                                                    with_commas(bytes).c_str(), with_commas(kTinyLlama.bytes_bf16).c_str()));
    chk.report(all_bf16, "every tensor is BF16");
  } else {
    chk.skipped("TinyLlama acceptance numbers (config is not TinyLlama-1.1B)");
  }

  if (a.has("quick")) {
    chk.skipped("full NaN/Inf scan of every weight (--quick)");
  } else {
    const auto t0 = std::chrono::steady_clock::now();
    std::uint64_t bad = 0, scanned = 0;
    std::string first_bad;
    for (const auto& t : st.tensors()) {
      if (!dtype_converts_to_f32(t.dtype)) continue;
      const auto raw = st.bytes(t);
      const std::size_t sz = dtype_size(t.dtype);
      for (std::uint64_t i = 0; i < t.numel; ++i) {
        bool nonfinite = false;
        if (sz == 2) {
          std::uint16_t h;
          std::memcpy(&h, raw.data() + 2 * i, 2);
          const std::uint16_t expmask = (t.dtype == DType::BF16) ? 0x7F80 : 0x7C00;
          nonfinite = (h & expmask) == expmask;
        } else {
          std::uint32_t u;
          std::memcpy(&u, raw.data() + 4 * i, 4);
          nonfinite = (u & 0x7F800000u) == 0x7F800000u;
        }
        if (nonfinite && bad++ == 0) first_bad = t.name + "[" + std::to_string(i) + "]";
      }
      scanned += t.numel;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    chk.report(bad == 0, strf("no NaN/Inf in %s weights (scanned %s in %.1f s)%s%s", with_commas(scanned).c_str(),
                              human_bytes(bytes).c_str(), s, bad ? "; first bad: " : "", first_bad.c_str()));
  }

  // BF16 decode spot check: a known-structure tensor (RMSNorm weights are
  // positive-ish, O(1)) decodes to finite, plausible floats.
  if (const TensorInfo* norm = st.find("model.layers.0.input_layernorm.weight")) {
    const TensorStats s = tensor_stats(st, *norm);
    chk.report(s.nan == 0 && s.inf == 0 && s.absmax < 100.0 && s.absmax > 1e-4,
               strf("BF16 decode sanity: layer-0 input_layernorm mean %.4g, absmax %.4g", s.mean, s.absmax));
  }

  if (a.has("golden")) {
    const fs::path probe = fs::path(a.get("golden")) / "weight_probe.json";
    if (fs::exists(probe)) {
      try {
        verify_weight_probe(st, probe, chk);
      } catch (const Error& e) {
        chk.report(false, std::string("golden weight probe: ") + e.what());
      }
    } else {
      chk.skipped("golden weight probe (" + probe.string() + " not found; run tools/dump_reference.py)");
    }
  } else {
    chk.skipped("golden weight probe (pass --golden tests/golden after running tools/dump_reference.py)");
  }

  std::printf("RESULT: %s  (%d passed, %d failed, %d skipped)\n", chk.fail ? "FAIL" : "PASS", chk.pass, chk.fail,
              chk.skip);
  return chk.fail ? 1 : 0;
}

// ------------------------------------------------------------------ membench / sysinfo

void print_system(const SystemInfo& s) {
  std::printf("CPU      : %s\n", s.cpu_model.c_str());
  std::printf("Cores    : %s physical / %d logical | OS %s\n",
              s.physical_cores > 0 ? std::to_string(s.physical_cores).c_str() : "?", s.logical_cpus, s.os.c_str());
  std::printf("RAM      : %s total, %s available\n", s.total_ram_bytes ? human_bytes(s.total_ram_bytes).c_str() : "?",
              s.avail_ram_bytes ? human_bytes(s.avail_ram_bytes).c_str() : "?");
  std::printf("Features : %s\n", cpu_feature_string(s).c_str());
  std::printf("Build    : %s | %s | flags: %s\n", s.compiler.c_str(), s.build_type.c_str(), s.build_flags.c_str());
}

std::vector<std::int32_t> parse_id_list(const std::string& s) {
  std::vector<std::int32_t> out;
  std::size_t pos = 0;
  while (pos <= s.size()) {
    const std::size_t comma = s.find(',', pos);
    const std::string item = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    char* end = nullptr;
    const long v = std::strtol(item.c_str(), &end, 10);
    if (item.empty() || *end != '\0' || v < 0) throw Error("--ids expects e.g. 1,529,29989; got '" + s + "'");
    out.push_back(static_cast<std::int32_t>(v));
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return out;
}

std::vector<int> parse_thread_list(const std::string& s) {
  std::vector<int> out;
  std::size_t pos = 0;
  while (pos <= s.size()) {
    const std::size_t comma = s.find(',', pos);
    const std::string item = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    char* end = nullptr;
    const long v = std::strtol(item.c_str(), &end, 10);
    if (item.empty() || *end != '\0' || v < 1) throw Error("--threads expects e.g. 1,2,4,8; got '" + s + "'");
    out.push_back(static_cast<int>(v));
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return out;
}

std::vector<int> sweep_threads(const SystemInfo& s) {
  const int logical = std::max(1, s.logical_cpus);
  std::set<int> t;
  for (int k = 1; k <= logical; k *= 2) t.insert(k);
  if (s.physical_cores > 0 && s.physical_cores <= logical) t.insert(s.physical_cores);
  t.insert(logical);
  return {t.begin(), t.end()};
}

int cmd_membench(const Args& a) {
  const SystemInfo sys = query_system_info();
  MembenchOptions o;
  const long long mb = a.get_int("mb", 1024);
  if (mb < 1) throw Error("--mb must be >= 1");
  o.buffer_bytes = static_cast<std::uint64_t>(mb) << 20;
  o.runs = static_cast<int>(a.get_int("runs", 5));
  o.passes = static_cast<int>(a.get_int("passes", 3));
  o.warmup_runs = static_cast<int>(a.get_int("warmup", 1));
  o.allow_low_memory = a.has("force");
  o.thread_counts = (a.has("threads") && !a.has("sweep")) ? parse_thread_list(a.get("threads")) : sweep_threads(sys);

  std::printf("EdgeForge %s - memory read bandwidth (STREAM-style, software-measured)\n", kVersion);
  print_system(sys);
  std::printf("Setup    : %s working set | %d timed runs x %d passes | %d warm-up run(s)\n",
              human_bytes(o.buffer_bytes).c_str(), o.runs, o.passes, o.warmup_runs);
  std::printf("\n  threads   best GB/s   median GB/s   worst GB/s   checksum\n");
  std::fflush(stdout);
  const auto results = run_membench(o, [](const MembenchResult& r) {
    std::printf("  %7d   %9.2f   %11.2f   %10.2f   %s\n", r.threads, r.best_gbps, r.median_gbps, r.worst_gbps,
                r.checksum_ok ? "OK" : "MISMATCH");
    std::fflush(stdout);
  });

  const MembenchResult* best = &results.front();
  bool all_ok = true;
  for (const auto& r : results) {
    if (r.median_gbps > best->median_gbps) best = &r;
    all_ok = all_ok && r.checksum_ok;
  }
  std::printf("\nPeak median read bandwidth: %.2f GB/s at %d threads  <- denominator for bandwidth efficiency\n",
              best->median_gbps, best->threads);
  std::printf("Note: achievable software read bandwidth, not a DRAM hardware counter; expect it below the\n"
              "      RAM's spec-sheet number. Run plugged in, performance power mode, apps closed.\n");
  if (!all_ok) std::printf("ERROR: checksum mismatch - results are not trustworthy.\n");

  if (a.has("json")) {
    const fs::path out = a.get("json");
    std::ofstream f(out, std::ios::binary);
    if (!f) throw Error("cannot write '" + out.string() + "'");
    f << "{\n  \"tool\": \"edgeforge membench\",\n  \"version\": \"" << json::escape(kVersion) << "\",\n";
    f << "  \"system\": {\"cpu\": \"" << json::escape(sys.cpu_model) << "\", \"physical_cores\": " << sys.physical_cores
      << ", \"logical_cpus\": " << sys.logical_cpus << ", \"total_ram_bytes\": " << sys.total_ram_bytes
      << ", \"os\": \"" << json::escape(sys.os) << "\", \"features\": \"" << json::escape(cpu_feature_string(sys))
      << "\", \"compiler\": \"" << json::escape(sys.compiler) << "\", \"build_type\": \""
      << json::escape(sys.build_type) << "\", \"build_flags\": \"" << json::escape(sys.build_flags) << "\"},\n";
    f << "  \"setup\": {\"buffer_bytes\": " << o.buffer_bytes << ", \"runs\": " << o.runs << ", \"passes\": "
      << o.passes << ", \"warmup_runs\": " << o.warmup_runs << "},\n  \"results\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
      const auto& r = results[i];
      f << "    {\"threads\": " << r.threads << ", \"best_gbps\": " << strf("%.4f", r.best_gbps)
        << ", \"median_gbps\": " << strf("%.4f", r.median_gbps) << ", \"worst_gbps\": " << strf("%.4f", r.worst_gbps)
        << ", \"checksum_ok\": " << (r.checksum_ok ? "true" : "false") << ", \"run_gbps\": [";
      for (std::size_t k = 0; k < r.run_gbps.size(); ++k) f << (k ? ", " : "") << strf("%.4f", r.run_gbps[k]);
      f << "]}" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    f << "  ],\n  \"peak_median_gbps\": " << strf("%.4f", best->median_gbps) << ",\n  \"peak_threads\": "
      << best->threads << "\n}\n";
    std::printf("Saved %s\n", out.string().c_str());
  }
  return all_ok ? 0 : 1;
}

// ---- Stage 7: threads and kernels for the engine commands

unsigned default_threads() {
  const SystemInfo si = query_system_info();
  if (si.physical_cores > 0) return static_cast<unsigned>(si.physical_cores);
  return std::max(1u, std::thread::hardware_concurrency());
}

// nullptr for 1 thread (the engine then runs on the calling thread only).
std::unique_ptr<ThreadPool> make_pool(const Args& a) {
  const long long t = a.has("threads") ? a.get_int("threads", 1) : static_cast<long long>(default_threads());
  if (t < 1 || t > 1024) throw Error("--threads expects 1..1024");
  return t > 1 ? std::make_unique<ThreadPool>(static_cast<unsigned>(t)) : nullptr;
}

std::string engine_label(const ThreadPool* pool) {
  const unsigned t = pool != nullptr ? pool->size() : 1u;
  return strf("%s kernels, %u thread%s", kern::backend_name(kern::backend()), t, t > 1 ? "s" : "");
}

std::vector<unsigned> parse_uint_list(const std::string& s) {
  std::vector<unsigned> v;
  for (const auto id : parse_id_list(s)) {
    if (id < 1) throw Error("--threads list entries must be >= 1");
    v.push_back(static_cast<unsigned>(id));
  }
  return v;
}

double membench_peak(const Args& a) {
  const std::string path = a.get("bench", fs::exists("bench/membench.json") ? "bench/membench.json" : "");
  if (path.empty()) return 0.0;
  std::ifstream in(path, std::ios::binary);
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (text.empty()) return 0.0;
  return json::parse(text).at("peak_median_gbps").as_double();
}

std::vector<std::int32_t> bench_prompt(const Args& a) {
  std::vector<std::int32_t> ids;
  if (a.has("ids-from")) {
    ids = GoldenPrompt::open(a.get("ids-from")).input_ids();
  } else if (a.has("ids")) {
    for (const auto id : parse_id_list(a.get("ids"))) ids.push_back(id);
  } else if (fs::exists("tests/golden/prompts/p0_factual/manifest.json")) {
    ids = GoldenPrompt::open("tests/golden/prompts/p0_factual").input_ids();
  }
  return ids;
}

int cmd_profile(const Args& a) {
  ProfileOptions o;
  o.qmodel = a.get("qmodel");
  o.model_dir = a.get("model-dir");
  o.prompt = bench_prompt(a);
  o.tokens = static_cast<std::size_t>(std::max(2LL, a.get_int("tokens", 32)));
  const auto pool = make_pool(a);
  o.pool = pool.get();
  o.peak_gbps = membench_peak(a);
  return run_profile(o);
}

int cmd_bench_decode(const Args& a) {
  BenchDecodeOptions o;
  o.qmodel = a.get("qmodel");
  o.model_dir = a.get("model-dir");
  o.prompt = bench_prompt(a);
  o.tokens = static_cast<std::size_t>(std::max(2LL, a.get_int("tokens", 32)));
  if (a.has("threads")) o.threads = parse_uint_list(a.get("threads"));
  o.ladder = !a.has("no-ladder");
  o.peak_gbps = membench_peak(a);
  return run_bench_decode(o);
}

// The BF16 reference (--model-dir) or the quantized engine on an .efq file (--qmodel).
Model load_engine_model(const Args& a) {
  if (a.has("qmodel") && a.has("model-dir")) throw Error("give either --model-dir (BF16) or --qmodel (.efq), not both");
  if (a.has("qmodel")) return Model::load_efq(a.get("qmodel"));
  if (!a.has("model-dir")) throw Error("--model-dir DIR (BF16) or --qmodel FILE.efq is required");
  return Model::load(a.get("model-dir"));
}

int cmd_generate(const Args& a) {
  std::vector<std::int32_t> ids;
  if (a.has("ids-from")) {
    ids = GoldenPrompt::open(a.get("ids-from")).input_ids();
  } else if (a.has("ids")) {
    for (const auto& item : parse_id_list(a.get("ids"))) ids.push_back(item);
  } else {
    throw Error("give the prompt as --ids 1,529,... or --ids-from tests/golden/prompts/<name>");
  }
  const auto t_load = std::chrono::steady_clock::now();
  const Model m = load_engine_model(a);
  const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();
  const auto max_new = static_cast<std::size_t>(std::max(1LL, a.get_int("max-new", 32)));
  const auto eos = static_cast<std::int32_t>(a.get_int("eos", m.config().eos_token_id));
  const auto pool = make_pool(a);
  Transformer tr(m, ids.size() + max_new, pool.get());
  std::printf("EdgeForge %s - generate (%s, FP32 activations)\n", kVersion, engine_label(pool.get()).c_str());
  std::printf("model: %s\n", m.description().c_str());
  std::printf("model loaded (mmap) in %.2f s | KV cache %s | %s of weights per token\n", load_s,
              human_bytes(tr.kv_cache_bytes()).c_str(), human_bytes(m.weight_bytes_per_token()).c_str());
  std::string p;
  for (auto id : ids) p += (p.empty() ? "" : " ") + std::to_string(id);
  std::printf("prompt ids (%zu): %s\n", ids.size(), p.c_str());
  const GreedyResult r = generate_greedy(tr, ids, max_new, eos);
  std::string g;
  for (auto id : r.tokens) g += (g.empty() ? "" : " ") + std::to_string(id);
  std::printf("generated (%zu)%s: %s\n", r.tokens.size(),
              (!r.tokens.empty() && r.tokens.back() == eos) ? ", ended on EOS" : "", g.c_str());
  std::printf("prefill: %zu tok in %.2f s (%.2f tok/s)\n", ids.size(), r.prefill_seconds,
              static_cast<double>(ids.size()) / r.prefill_seconds);
  if (r.decode_forwards > 0) {
    const double tps = static_cast<double>(r.decode_forwards) / r.decode_seconds;
    const double gbps = tps * static_cast<double>(m.weight_bytes_per_token()) / 1e9;
    std::printf("decode : %zu tok in %.2f s (%.2f tok/s) -> %.2f GB/s of weights streamed", r.decode_forwards,
                r.decode_seconds, tps, gbps);
    if (a.has("bench")) {
      std::ifstream in(a.get("bench"), std::ios::binary);
      const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      if (!text.empty()) {
        const double peak = json::parse(text).at("peak_median_gbps").as_double();
        std::printf(" = %.0f%% of the membench baseline %.2f GB/s", 100.0 * gbps / peak, peak);
      }
    }
    std::printf("\n");
  }
  std::printf("peak RSS %s (weights are memory-mapped)\n", human_bytes(peak_rss_bytes()).c_str());
  return 0;
}

int cmd_tokenize(const Args& a) {
  if (!a.has("tokenizer-dir") || a.has("text") == a.has("text-file")) {
    throw Error("tokenize needs --tokenizer-dir DIR and one of --text TEXT / --text-file FILE");
  }
  const Tokenizer tok = Tokenizer::load(a.get("tokenizer-dir"));
  if (a.has("text-file")) {  // Stage 8: whole files (e.g. wikitext-2), ids written to --ids-out
    std::ifstream in(a.get("text-file"), std::ios::binary);
    if (!in) throw Error("cannot open '" + a.get("text-file") + "'");
    const std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto t0 = std::chrono::steady_clock::now();
    const auto ids = tok.encode(body, true, !a.has("no-special"));
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%s: %zu bytes -> %zu tokens (BOS added) in %.2f s\n", a.get("text-file").c_str(), body.size(),
                ids.size(), s);
    if (a.has("ids-out")) {
      std::ofstream out(a.get("ids-out"), std::ios::binary);
      for (auto id : ids) out << id << '\n';
      if (!out) throw Error("cannot write '" + a.get("ids-out") + "'");
      std::printf("ids written to %s\n", a.get("ids-out").c_str());
    }
    return 0;
  }
  const auto ids = tok.encode(a.get("text"), true, !a.has("no-special"));
  std::printf("%zu tokens (BOS added):\n", ids.size());
  for (auto id : ids) std::printf("  %6d  %s\n", id, json::escape(tok.piece(id)).c_str());
  std::printf("decoded back: \"%s\"\n", json::escape(tok.decode(ids, true)).c_str());
  return 0;
}

int cmd_chat(const Args& a) {
  if (a.has("qmodel") && !a.has("tokenizer-dir") && !a.has("model-dir")) {
    throw Error("chat --qmodel needs --tokenizer-dir DIR (the folder with tokenizer.json)");
  }
  const fs::path model_dir = a.get("model-dir");
  const Model model = a.has("qmodel") ? Model::load_efq(a.get("qmodel")) : load_engine_model(a);
  const Tokenizer tok = Tokenizer::load(a.has("tokenizer-dir") ? fs::path(a.get("tokenizer-dir")) : model_dir);
  ChatOptions o;
  o.max_seq = static_cast<std::size_t>(std::max(16LL, a.get_int("ctx", model.config().max_position_embeddings)));
  o.max_new = static_cast<std::size_t>(std::max(1LL, a.get_int("max-new", 256)));
  o.system_prompt = a.has("no-system") ? std::string() : a.get("system", kDefaultSystemPrompt);
  if (a.has("no-stop-markers")) o.stop.stop_strings.clear();
  if (a.has("stop")) {
    if (a.get("stop").empty()) throw Error("--stop needs a non-empty text");
    o.stop.stop_strings.push_back(a.get("stop"));
  }
  o.stop.repeat_count = static_cast<std::size_t>(std::max(0LL, a.get_int("repeat-count", 3)));
  o.stop.repeat_span = static_cast<std::size_t>(std::max(1LL, a.get_int("repeat-span", 12)));
  o.stop.eos_min_p = static_cast<float>(a.get_double("eos-min-p", 0.0));
  if (o.stop.eos_min_p < 0.0f || o.stop.eos_min_p > 1.0f) throw Error("--eos-min-p expects 0..1");
  o.sampler.temperature = static_cast<float>(a.get_double("temp", 0.0));
  o.sampler.top_k = static_cast<int>(a.get_int("top-k", 0));
  o.sampler.top_p = static_cast<float>(a.get_double("top-p", 1.0));
  o.sampler.repetition_penalty = static_cast<float>(a.get_double("rep-penalty", 1.0));
  o.sampler.seed = static_cast<std::uint64_t>(a.get_int("seed", 0));
  const auto pool = make_pool(a);
  o.pool = pool.get();
  ChatSession chat(model, tok, o);

  auto print_stats = [](const TurnStats& s) {
    std::string end = stop_reason_name(s.reason);
    if (s.reason == StopReason::StopString) end += " '" + s.stop_detail + "', reply cut before it";
    if (s.reason == StopReason::Repetition) end += " (" + s.stop_detail + "), reply cut at the first repeat";
    if (s.reason == StopReason::MaxNew) end += " (--max-new)";
    std::printf("  [prompt %zu tok, %zu reused, %zu computed in %.1f s | reply %zu tok in %.1f s = %.2f tok/s | end: %s%s]\n",
                s.prompt_tokens, s.reused_tokens, s.prefill_tokens, s.prefill_s, s.generated, s.decode_s,
                s.decode_s > 0 ? static_cast<double>(s.generated) / s.decode_s : 0.0, end.c_str(),
                s.dropped_messages ? ", oldest turns dropped" : "");
  };
  auto stream = [](const std::string& t) {
    std::fwrite(t.data(), 1, t.size(), stdout);
    std::fflush(stdout);
  };
  if (a.has("prompt")) {  // one-shot
    TurnStats s;
    chat.reply(a.get("prompt"), stream, &s);
    std::printf("\n");
    print_stats(s);
    return 0;
  }
  const std::string sampling = o.sampler.temperature <= 0 ? std::string("greedy")
                                                          : strf("temp %.2f, top-k %d, top-p %.2f, seed %llu",
                                                                 static_cast<double>(o.sampler.temperature), o.sampler.top_k,
                                                                 static_cast<double>(o.sampler.top_p),
                                                                 static_cast<unsigned long long>(o.sampler.seed));
  std::printf("EdgeForge %s - CPU inference engine, no ML libraries\n", kVersion);
  std::printf("model: %s (%lld layers, %s, %s) | tokenizer: %zu tokens%s\n",
              (a.has("qmodel") ? fs::path(a.get("qmodel")) : model_dir).filename().string().c_str(),
              static_cast<long long>(model.config().num_hidden_layers), model.description().c_str(),
              engine_label(pool.get()).c_str(),
              tok.vocab_size(), tok.chat_template().empty() || is_zephyr_template(tok.chat_template()) ? "" : " (unknown chat template!)");
  std::printf("sampling: %s | context %zu tokens | commands: /reset  /exit\n", sampling.c_str(), o.max_seq);
  std::printf("system: %s\n\n", o.system_prompt.empty() ? "(none)" : ("\"" + o.system_prompt + "\"").c_str());
  std::string line;
  while (true) {
    std::printf("You: ");
    std::fflush(stdout);
    if (!std::getline(std::cin, line)) break;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    if (line == "/exit" || line == "/quit") break;
    if (line == "/reset") {
      chat.reset();
      std::printf("(conversation cleared)\n\n");
      continue;
    }
    std::printf("AI: ");
    std::fflush(stdout);
    TurnStats s;
    try {
      chat.reply(line, stream, &s);
    } catch (const Error& e) {
      std::printf("\n  error: %s\n\n", e.what());
      continue;
    }
    std::printf("\n");
    print_stats(s);
    std::printf("\n");
  }
  return 0;
}

QType parse_qtype_arg(const Args& a, const std::string& key, QType def) {
  if (!a.has(key)) return def;
  const auto t = qtype_from_string(a.get(key));
  if (!t) throw Error("--" + key + " expects one of q4_0, q8_0, bf16, f16, f32; got '" + a.get(key) + "'");
  return *t;
}

int cmd_quantize(const Args& a) {
  if (!a.has("model-dir") || !a.has("out")) throw Error("quantize needs --model-dir DIR and --out FILE.efq");
  QuantizeOptions o;
  o.layers = parse_qtype_arg(a, "layers", QType::Q4_0);
  o.lm_head = parse_qtype_arg(a, "lm-head", QType::Q8_0);
  o.embed = parse_qtype_arg(a, "embed", QType::BF16);
  const ModelConfig cfg = load_model_config(std::filesystem::path(a.get("model-dir")) / "config.json");
  const PolicySelection sel = select_policy(a.get("policy", "auto"), cfg);
  o.overrides = sel.overrides;
  o.policy_name = sel.name;
  if (a.has("override")) {
    std::string spec = a.get("override");
    for (char& ch : spec) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (spec == "none") throw Error("--override none is gone: use --policy uniform for the no-exception baseline");
    const auto extra = parse_overrides(a.get("override"));
    o.overrides.insert(o.overrides.end(), extra.begin(), extra.end());
    o.policy_name += "+custom";
  }
  o.search = !a.has("no-search");
  o.threads = static_cast<unsigned>(std::max(0LL, a.get_int("threads", 0)));
  std::printf("EdgeForge %s - quantize\n  layers %s | lm_head %s | embed %s | norms F32 | block 32 | scale %s\n",
              kVersion, qtype_name(o.layers), qtype_name(o.lm_head), qtype_name(o.embed),
              o.search ? "MSE search" : "absmax");
  std::printf("  policy   : %s (%s)\n", o.policy_name.c_str(), sel.note.c_str());
  if (o.overrides.empty()) std::printf("  overrides: none\n");
  for (const auto& ov : o.overrides) std::printf("  override : %s -> %s\n", ov.pattern.c_str(), qtype_name(ov.type));
  double layer_worst = 0.0;
  auto t_layer = std::chrono::steady_clock::now();
  const auto res = quantize_model(a.get("model-dir"), a.get("out"), o, [&](const QTensor& t, std::size_t i, std::size_t n) {
    (void)i;
    (void)n;
    if (qtype_is_block(t.type)) layer_worst = std::max(layer_worst, t.rel_rmse);
    const bool layer_end = t.name.find("post_attention_layernorm") != std::string::npos;
    if (layer_end || t.name.rfind("model.layers.", 0) != 0) {
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_layer).count();
      if (layer_end) {
        std::printf("  %-34s worst rel RMSE %.4f  [%.1f s]\n", t.name.substr(0, t.name.find(".post_")).c_str(),
                    layer_worst, s);
        layer_worst = 0.0;
      } else {
        std::printf("  %-34s %-5s rel RMSE %.4f  [%.1f s]\n", t.name.c_str(), qtype_name(t.type), t.rel_rmse, s);
      }
      std::fflush(stdout);
      t_layer = std::chrono::steady_clock::now();
    }
  });
  for (const auto& note : res.notes) std::printf("  note: %s\n", note.c_str());
  std::printf("\nwrote %s in %.1f s\n", a.get("out").c_str(), res.seconds);
  if (!o.overrides.empty()) {
    const auto x = res.override_extra_bytes;
    std::printf("  overrides   : %s%s of tensor data vs the uniform policy\n", x < 0 ? "-" : "+",
                human_bytes(static_cast<std::uint64_t>(x < 0 ? -x : x)).c_str());
  }
  std::printf("  size        : %s -> %s (%.2fx smaller)\n", human_bytes(res.source_tensor_bytes).c_str(),
              human_bytes(res.file_bytes).c_str(),
              static_cast<double>(res.source_tensor_bytes) / static_cast<double>(res.file_bytes));
  std::printf("  read/token  : %s -> %s (%.2fx less memory traffic per decoded token)\n",
              human_bytes(res.source_bytes_per_token).c_str(), human_bytes(res.bytes_per_token).c_str(),
              static_cast<double>(res.source_bytes_per_token) / static_cast<double>(res.bytes_per_token));
  std::printf("next: edgeforge verify-quant --qmodel %s --model-dir %s --golden tests/golden\n", a.get("out").c_str(),
              a.get("model-dir").c_str());
  return 0;
}

int cmd_sysinfo() {
  std::printf("EdgeForge %s - system\n", kVersion);
  print_system(query_system_info());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
  SetConsoleOutputCP(CP_UTF8);  // Hindi / emoji in the Windows console
  SetConsoleCP(CP_UTF8);
#endif
  try {
    const Args a = parse_args(argc, argv);
    if (a.has("kernels")) kern::set_backend(kern::backend_from_string(a.get("kernels")));
    if (a.command.empty() || a.command == "help" || a.command == "--help" || a.command == "-h") {
      print_usage();
      return a.command.empty() ? 2 : 0;
    }
    if (a.command == "--version" || a.command == "version") {
      std::printf("EdgeForge %s\n", kVersion);
      return 0;
    }
    if (a.command == "inspect") return cmd_inspect(a);
    if (a.command == "verify") return cmd_verify(a);
    if (a.command == "membench") return cmd_membench(a);
    if (a.command == "sysinfo") return cmd_sysinfo();
    if (a.command == "verify-forward") {
      if (!a.has("golden") || !a.has("model-dir")) throw Error("verify-forward needs --golden DIR and --model-dir DIR");
      VerifyForwardOptions o;
      o.golden_dir = a.get("golden");
      o.model_dir = a.get("model-dir");
      o.prompt = a.get("prompt");
      std::printf("(engine kernels: %s)\n", kern::backend_name(kern::backend()));
      return run_verify_forward(o);
    }
    if (a.command == "generate") return cmd_generate(a);
    if (a.command == "bench-decode") return cmd_bench_decode(a);
    if (a.command == "profile") return cmd_profile(a);
    if (a.command == "chat") return cmd_chat(a);
    if (a.command == "quantize") return cmd_quantize(a);
    if (a.command == "verify-quant") {
      if (!a.has("qmodel") || !a.has("model-dir")) throw Error("verify-quant needs --qmodel FILE and --model-dir DIR");
      VerifyQuantOptions o;
      o.qmodel = a.get("qmodel");
      o.model_dir = a.get("model-dir");
      o.golden_dir = a.get("golden");
      o.threads = static_cast<unsigned>(std::max(0LL, a.get_int("threads", 0)));
      return run_verify_quant(o);
    }
    if (a.command == "eval-quant") {
      if (!a.has("qmodel")) throw Error("eval-quant needs --qmodel FILE.efq");
      EvalQuantOptions o;
      o.qmodel = a.get("qmodel");
      o.model_dir = a.get("model-dir");
      o.golden_dir = a.get("golden");
      o.text_file = a.get("text");
      o.tokenizer_dir = a.get("tokenizer-dir");
      o.max_text_tokens = static_cast<std::size_t>(std::max(2LL, a.get_int("max-tokens", 128)));
      o.window = static_cast<std::size_t>(std::max(0LL, a.get_int("window", 0)));
      o.max_chunks = static_cast<std::size_t>(std::max(0LL, a.get_int("chunks", 0)));
      const auto pool = make_pool(a);
      o.pool = pool.get();
      return run_eval_quant(o);
    }
    if (a.command == "quant-analyze") {
      if (!a.has("model-dir")) throw Error("quant-analyze needs --model-dir DIR");
      QuantAnalyzeOptions o;
      o.model_dir = a.get("model-dir");
      o.golden_dir = a.get("golden");
      return run_quant_analyze(o);
    }
    if (a.command == "tokenize") return cmd_tokenize(a);
    if (a.command == "eos-study") {
      EosStudyOptions o;
      o.golden = a.get("golden");
      o.model_dir = a.get("model-dir");
      o.qmodel = a.get("qmodel");
      o.tokenizer_dir = a.get("tokenizer-dir");
      o.results = a.get("results");
      o.labels = a.get("labels");
      o.out = a.get("out", "bench/eos_study");
      const auto pool = make_pool(a);
      o.pool = pool.get();
      return run_eos_study(o);
    }
    if (a.command == "verify-chat") {
      if (!a.has("golden")) throw Error("verify-chat needs --golden DIR (with chat_generation.json)");
      VerifyChatOptions o;
      o.golden_dir = a.get("golden");
      o.model_dir = a.get("model-dir");
      o.qmodel = a.get("qmodel");
      o.tokenizer_dir = a.get("tokenizer-dir");
      if (!o.qmodel.empty() && !o.model_dir.empty()) {
        throw Error("verify-chat runs one engine: --model-dir DIR (BF16) or --qmodel FILE.efq --tokenizer-dir DIR");
      }
      const auto pool = make_pool(a);
      o.pool = pool.get();
      return run_verify_chat(o);
    }
    if (a.command == "verify-tokenizer") {
      if (!a.has("golden") || !a.has("tokenizer-dir")) throw Error("verify-tokenizer needs --tokenizer-dir DIR and --golden DIR");
      VerifyTokenizerOptions o;
      o.golden_dir = a.get("golden");
      o.tokenizer_dir = a.get("tokenizer-dir");
      o.model_dir = a.get("model-dir");
      return run_verify_tokenizer(o);
    }
    if (a.command == "verify-math") {
      if (!a.has("golden")) throw Error("--golden DIR is required (folder written by tools/dump_reference.py)");
      VerifyMathOptions o;
      o.golden_dir = a.get("golden");
      o.model_dir = a.get("model-dir");
      o.bench_json = a.get("bench");
      o.prompt = a.get("prompt", "p0_factual");
      return run_verify_math(o);
    }
    print_usage();
    return 2;
  } catch (const Error& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "internal error: %s\n", e.what());
    return 1;
  }
}
