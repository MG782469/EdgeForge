// EdgeForge - Stage 3: TinyLlama (Llama-family) FP32 reference forward pass.
//
// Model       : zero-copy view of the BF16 weights inside the memory-mapped
//               model.safetensors (+ norm weights widened to f32).
// Transformer : one-token-at-a-time forward with a KV cache. Prefill = feeding
//               the prompt tokens at positions 0..S-1; decode = feeding each new
//               token at the next position. Built only from the Stage 2 kernels.
//
// Everything is scalar FP32 with FP32 accumulation; this is the reference that
// the quantized / threaded / SIMD engine of later stages is tested against.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "ef/kernels.h"
#include "ef/model_config.h"
#include "ef/safetensors.h"

namespace ef {

struct LayerWeights {
  const std::uint16_t* wq = nullptr;  // [D][D]
  const std::uint16_t* wk = nullptr;  // [kv_dim][D]
  const std::uint16_t* wv = nullptr;  // [kv_dim][D]
  const std::uint16_t* wo = nullptr;  // [D][D]
  const std::uint16_t* w_gate = nullptr;  // [F][D]
  const std::uint16_t* w_up = nullptr;    // [F][D]
  const std::uint16_t* w_down = nullptr;  // [D][F]
  std::vector<float> attn_norm;  // input_layernorm [D]
  std::vector<float> ffn_norm;   // post_attention_layernorm [D]
};

class Model {
 public:
  // Loads config.json + model.safetensors from `model_dir`; validates every
  // tensor name/shape against the config. Weights stay memory-mapped.
  static Model load(const std::filesystem::path& model_dir);

  const ModelConfig& config() const noexcept { return cfg_; }
  const SafetensorsFile& file() const noexcept { return st_; }
  const std::uint16_t* embed_tokens() const noexcept { return embed_; }
  const std::uint16_t* lm_head() const noexcept { return lm_head_; }
  const std::vector<float>& final_norm() const noexcept { return final_norm_; }
  const LayerWeights& layer(std::size_t l) const { return layers_.at(l); }
  std::size_t n_layers() const noexcept { return layers_.size(); }

  // Bytes of weights streamed from memory per decoded token: every layer
  // matrix + norms + lm_head (the embedding table contributes a single row).
  std::uint64_t weight_bytes_per_token() const noexcept { return bytes_per_token_; }

 private:
  ModelConfig cfg_;
  SafetensorsFile st_;
  const std::uint16_t* embed_ = nullptr;
  const std::uint16_t* lm_head_ = nullptr;
  std::vector<float> final_norm_;
  std::vector<LayerWeights> layers_;
  std::uint64_t bytes_per_token_ = 0;
};

// Optional capture of intermediate results for verification.
struct ForwardTrace {
  bool capture = false;
  std::vector<float> layer_out;   // [n_layers * D]: residual stream after each layer
  std::vector<float> final_norm;  // [D]: final RMSNorm output (lm_head input)
};

class Transformer {
 public:
  // max_seq: KV-cache capacity in tokens (<= max_position_embeddings).
  Transformer(const Model& model, std::size_t max_seq);

  // Runs `token` at position `pos`. Requires pos <= cached() (pos < cached()
  // rewinds: the cache is truncated to pos and overwritten) and pos < max_seq().
  // Returns the next-token logits [vocab], or nullptr if compute_logits is false
  // (prefill positions whose logits are not needed skip the lm_head GEMV).
  const float* forward(std::int32_t token, std::size_t pos, bool compute_logits = true, ForwardTrace* trace = nullptr);

  std::size_t cached() const noexcept { return cached_; }
  std::size_t max_seq() const noexcept { return max_seq_; }
  void reset() noexcept { cached_ = 0; }
  const Model& model() const noexcept { return m_; }
  std::size_t kv_cache_bytes() const noexcept { return (k_cache_.size() + v_cache_.size()) * sizeof(float); }

 private:
  const Model& m_;
  std::size_t max_seq_ = 0, cached_ = 0;
  std::size_t D_ = 0, F_ = 0, H_ = 0, KV_ = 0, hd_ = 0, kvd_ = 0, V_ = 0, L_ = 0;
  float eps_ = 1e-5f;
  kern::RopeTable rope_;
  std::vector<float> k_cache_, v_cache_;  // [layer][max_seq][kv_dim]
  std::vector<float> x_, h_, q_, ctx_, o_, gate_, up_, scores_, logits_;
};

struct GreedyResult {
  std::vector<std::int32_t> tokens;           // generated ids (includes EOS if hit)
  std::vector<std::vector<float>> step_logits;  // logits after feeding each generated token (optional)
  double prefill_seconds = 0.0, decode_seconds = 0.0;
  std::size_t decode_forwards = 0;            // forward() calls in the decode phase
};

// Prefill `prompt` from position 0, then pick argmax tokens until `max_new`
// tokens or `eos_id`. keep_step_logits stores logits after each decode forward.
GreedyResult generate_greedy(Transformer& t, const std::vector<std::int32_t>& prompt, std::size_t max_new,
                             std::int32_t eos_id, std::size_t keep_step_logits = 0);

}  // namespace ef
