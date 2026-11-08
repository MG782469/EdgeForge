// EdgeForge - Stage 6: `edgeforge eval-quant`, the quantized engine end to end.
//
// Runs the full 22-layer engine on the packed .efq weights (no dequantized copy) and measures:
//   1. kernel correctness on the real weights: packed Q4_0 / Q8_0 dot = dequantize-then-dot;
//   2. next-token distributions vs the PyTorch FP32 reference (golden, teacher-forced):
//      KL(ref || ours), top-1 agreement, and the greedy continuation of every golden prompt;
//   3. perplexity of a text, our BF16 engine vs the quantized engine, with per-position KL - one
//      window, or chunks scored exactly like llama.cpp's llama-perplexity (comparable numbers);
//   4. speed (tok/s, GB/s of weights) of both engines and the quantized engine's peak RSS.
// Quality targets are fixed in advance (eval_quant.cpp) and never tuned to a result.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ef {

class ThreadPool;

struct EvalQuantReport {
  int passed = 0, failed = 0;
  std::vector<std::string> failed_checks;
  std::size_t kernel_rows = 0;
  double kernel_max_rel = 0.0;
  // Teacher-forced next-token distributions (golden positions + text positions).
  std::size_t positions = 0, top1 = 0, confident = 0, confident_top1 = 0, nonfinite = 0;
  double kl_mean = 0.0, kl_max = 0.0;
  std::size_t golden_positions = 0, text_positions = 0;
  // Greedy continuation of each golden prompt (free-running).
  std::size_t greedy_prompts = 0, greedy_identical = 0;
  // Text perplexity (0 = not measured).
  double ppl_ref = 0.0, ppl_q = 0.0;
  // Speed: forward passes per second (teacher-forced, logits at every position).
  double tps_ref = 0.0, tps_q = 0.0;
  std::uint64_t q_peak_rss = 0;  // process peak RSS before the BF16 reference is loaded
};

struct EvalQuantOptions {
  std::filesystem::path qmodel;         // the .efq file
  std::filesystem::path model_dir;      // BF16 checkpoint: text reference + speed baseline (optional)
  std::filesystem::path golden_dir;     // PyTorch FP32 reference prompts (optional)
  std::filesystem::path text_file;      // perplexity text (optional)
  std::filesystem::path tokenizer_dir;  // default: model_dir
  std::size_t max_text_tokens = 128;   // window == 0: one window of at most this many tokens
  // window > 0: llama-perplexity scheme - chunks of `window` tokens (first token -> BOS), only the
  // second half of each chunk scored; at most max_chunks chunks (0 = all that fit).
  std::size_t window = 0, max_chunks = 0;
  std::vector<std::int32_t> text_ids;   // tests: token ids to use instead of text_file
  ThreadPool* pool = nullptr;           // Stage 7: run both engines multi-threaded
  EvalQuantReport* report = nullptr;
};

int run_eval_quant(const EvalQuantOptions& o);

}  // namespace ef
