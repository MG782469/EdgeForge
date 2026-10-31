// EdgeForge — Llama-family model configuration (config.json) and checkpoint checks.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "ef/json.h"
#include "ef/safetensors.h"

namespace ef {

struct ModelConfig {
  std::string model_type;                  // must be "llama"
  std::vector<std::string> architectures;  // e.g. LlamaForCausalLM
  std::string hidden_act;                  // must be "silu"
  std::string torch_dtype;                 // informational ("bfloat16")

  std::int64_t hidden_size = 0;              // D
  std::int64_t intermediate_size = 0;        // MLP hidden (F)
  std::int64_t num_hidden_layers = 0;        // L
  std::int64_t num_attention_heads = 0;      // H
  std::int64_t num_key_value_heads = 0;      // KV heads (GQA)
  std::int64_t head_dim = 0;                 // D / H
  std::int64_t vocab_size = 0;               // V
  std::int64_t max_position_embeddings = 0;  // context length

  double rms_norm_eps = 0.0;
  double rope_theta = 10000.0;
  bool tie_word_embeddings = false;
  bool attention_bias = false;
  bool mlp_bias = false;
  std::int64_t bos_token_id = -1;
  std::int64_t eos_token_id = -1;

  std::int64_t kv_dim() const noexcept { return num_key_value_heads * head_dim; }
  std::int64_t gqa_group() const noexcept { return num_attention_heads / num_key_value_heads; }
};

// Parses and validates config.json. Throws ef::Error naming the bad field.
ModelConfig parse_model_config(const json::Value& root);
ModelConfig load_model_config(const std::filesystem::path& config_json);

struct ExpectedTensor {
  std::string name;
  std::vector<std::uint64_t> shape;
};

// The exact tensor list a Hugging Face Llama checkpoint must contain, in
// canonical order: embed, then per layer (q,k,v,o, gate,up,down, 2 norms),
// final norm, lm_head (unless tied).
std::vector<ExpectedTensor> expected_tensors(const ModelConfig& cfg);
std::uint64_t expected_param_count(const ModelConfig& cfg);

struct CheckReport {
  std::vector<std::string> errors;
  std::vector<std::string> warnings;
  std::uint64_t matched_tensors = 0;
  std::uint64_t matched_params = 0;
  bool ok() const noexcept { return errors.empty(); }
};

// Compares a checkpoint's tensors against what the config says must be there:
// every expected tensor present with the right shape and a float dtype, and
// nothing unexpected (rotary inv_freq buffers are ignored with a warning).
CheckReport check_checkpoint(const ModelConfig& cfg, const std::vector<TensorInfo>& tensors);

}  // namespace ef
