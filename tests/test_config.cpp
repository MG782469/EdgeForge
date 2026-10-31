// EdgeForge — config.json parsing, expected-tensor table and checkpoint checks.
#include <string>
#include <vector>

#include "ef/model_config.h"
#include "test_framework.h"
#include "test_util.h"

using namespace ef;

namespace {

// Verbatim structure of TinyLlama/TinyLlama-1.1B-Chat-v1.0 config.json.
const std::string kTinyLlamaConfig = R"({
  "architectures": ["LlamaForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 5632,
  "max_position_embeddings": 2048,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 22,
  "num_key_value_heads": 4,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.35.0",
  "use_cache": true,
  "vocab_size": 32000
})";

std::string replaced(const std::string& from, const std::string& to) {
  std::string s = kTinyLlamaConfig;
  const auto pos = s.find(from);
  if (pos == std::string::npos) throw std::runtime_error("test bug: '" + from + "' not in config");
  s.replace(pos, from.size(), to);
  return s;
}

ModelConfig parse_cfg(const std::string& text) { return parse_model_config(json::parse(text)); }

ModelConfig tiny_cfg() {
  return parse_cfg(R"({"model_type":"llama","hidden_size":8,"intermediate_size":16,"num_hidden_layers":2,
    "num_attention_heads":2,"num_key_value_heads":1,"vocab_size":10,"rms_norm_eps":1e-5})");
}

std::vector<TensorInfo> tensors_from(const ModelConfig& c) {
  std::vector<TensorInfo> out;
  std::uint64_t off = 0;
  for (const auto& e : expected_tensors(c)) {
    TensorInfo t;
    t.name = e.name;
    t.dtype = DType::BF16;
    t.shape = e.shape;
    t.numel = 1;
    for (auto d : e.shape) t.numel *= d;
    t.begin = off;
    t.end = off + 2 * t.numel;
    off = t.end;
    out.push_back(t);
  }
  return out;
}

}  // namespace

TEST(config_tinyllama_fields) {
  const ModelConfig c = parse_cfg(kTinyLlamaConfig);
  CHECK_EQ(c.model_type, std::string("llama"));
  CHECK_EQ(c.architectures.at(0), std::string("LlamaForCausalLM"));
  CHECK_EQ(c.hidden_size, 2048);
  CHECK_EQ(c.intermediate_size, 5632);
  CHECK_EQ(c.num_hidden_layers, 22);
  CHECK_EQ(c.num_attention_heads, 32);
  CHECK_EQ(c.num_key_value_heads, 4);
  CHECK_EQ(c.head_dim, 64);
  CHECK_EQ(c.kv_dim(), 256);
  CHECK_EQ(c.gqa_group(), 8);
  CHECK_EQ(c.vocab_size, 32000);
  CHECK_EQ(c.max_position_embeddings, 2048);
  CHECK_EQ(c.rms_norm_eps, 1e-05);
  CHECK_EQ(c.rope_theta, 10000.0);
  CHECK_EQ(c.tie_word_embeddings, false);
  CHECK_EQ(c.bos_token_id, 1);
  CHECK_EQ(c.eos_token_id, 2);
  CHECK_EQ(c.torch_dtype, std::string("bfloat16"));
}

TEST(config_tinyllama_acceptance_numbers) {
  // The hard numbers `edgeforge verify` checks the real checkpoint against.
  const ModelConfig c = parse_cfg(kTinyLlamaConfig);
  const auto t = expected_tensors(c);
  CHECK_EQ(t.size(), std::size_t{201});
  CHECK_EQ(expected_param_count(c), std::uint64_t{1'100'048'384});
  CHECK_EQ(expected_param_count(c) * 2, std::uint64_t{2'200'096'768});  // BF16 bytes
  CHECK_EQ(t.front().name, std::string("model.embed_tokens.weight"));
  CHECK_EQ(t.back().name, std::string("lm_head.weight"));
  CHECK(t[1].name == "model.layers.0.self_attn.q_proj.weight" && t[1].shape == (std::vector<std::uint64_t>{2048, 2048}));
  CHECK(t[2].name == "model.layers.0.self_attn.k_proj.weight" && t[2].shape == (std::vector<std::uint64_t>{256, 2048}));
  CHECK(t[7].name == "model.layers.0.mlp.down_proj.weight" && t[7].shape == (std::vector<std::uint64_t>{2048, 5632}));
  CHECK(t[9].name == "model.layers.0.post_attention_layernorm.weight" && t[9].shape == (std::vector<std::uint64_t>{2048}));
  CHECK(t[t.size() - 2].name == "model.norm.weight");
}

TEST(config_tied_embeddings_drop_lm_head) {
  const ModelConfig c = parse_cfg(replaced("\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true"));
  CHECK_EQ(expected_tensors(c).size(), std::size_t{200});
  CHECK_EQ(expected_param_count(c), std::uint64_t{1'100'048'384 - 65'536'000});
}

TEST(config_rejects_invalid) {
  CHECK_THROWS(parse_cfg(replaced("\"num_key_value_heads\": 4", "\"num_key_value_heads\": 5")),
               "not divisible by num_key_value_heads");
  CHECK_THROWS(parse_cfg(replaced("\"hidden_size\": 2048", "\"hidden_size\": 2000")),
               "not divisible by num_attention_heads");
  CHECK_THROWS(parse_cfg(replaced("\"rope_scaling\": null", "\"rope_scaling\": {\"type\": \"linear\", \"factor\": 2.0}")),
               "rope_scaling");
  CHECK_THROWS(parse_cfg(replaced("\"hidden_size\": 2048,", "")), "missing required field 'hidden_size'");
  CHECK_THROWS(parse_cfg(replaced("\"hidden_size\": 2048", "\"hidden_size\": \"2048\"")), "must be an integer");
  CHECK_THROWS(parse_cfg(replaced("\"hidden_size\": 2048", "\"hidden_size\": 2048.5")), "must be an integer, got 2048.5");
  CHECK_THROWS(parse_cfg(replaced("\"model_type\": \"llama\"", "\"model_type\": \"mistral\"")), "not supported");
  CHECK_THROWS(parse_cfg(replaced("\"num_attention_heads\": 32", "\"num_attention_heads\": 0")), "outside");
  CHECK_THROWS(parse_cfg(replaced("\"rms_norm_eps\": 1e-05", "\"rms_norm_eps\": 0")), "rms_norm_eps");
  CHECK_THROWS(parse_cfg(replaced("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"")), "hidden_act");
  CHECK_THROWS(parse_cfg(replaced("\"attention_bias\": false", "\"attention_bias\": true")), "attention_bias");
  CHECK_THROWS(parse_cfg(replaced("\"tie_word_embeddings\": false", "\"tie_word_embeddings\": 0")), "true or false");
  CHECK_THROWS(parse_cfg("[1, 2]"), "must be a JSON object");
}

TEST(config_check_checkpoint) {
  const ModelConfig c = tiny_cfg();
  CHECK_EQ(c.head_dim, 4);
  const auto good = tensors_from(c);
  CHECK_EQ(good.size(), std::size_t{21});  // 1 + 2*9 + 1 + 1
  auto r = check_checkpoint(c, good);
  CHECK(r.ok());
  CHECK_EQ(r.matched_tensors, std::uint64_t{21});
  CHECK_EQ(r.matched_params, expected_param_count(c));

  auto missing = good;
  missing.erase(missing.begin() + 3);  // drop layers.0 v_proj
  r = check_checkpoint(c, missing);
  CHECK(!r.ok());
  CHECK(r.errors.at(0).find("missing tensor 'model.layers.0.self_attn.v_proj.weight'") != std::string::npos);

  auto wrong_shape = good;
  wrong_shape[2].shape = {8, 8};  // k_proj should be [4, 8] (1 KV head * head_dim 4)
  r = check_checkpoint(c, wrong_shape);
  CHECK(!r.ok() && r.errors.at(0).find("expected [4, 8]") != std::string::npos);

  auto extra = good;
  TensorInfo junk;
  junk.name = "model.layers.0.self_attn.q_proj.bias";
  extra.push_back(junk);
  r = check_checkpoint(c, extra);
  CHECK(!r.ok() && r.errors.at(0).find("unexpected tensor") != std::string::npos);

  auto with_buffer = good;
  TensorInfo inv;
  inv.name = "model.layers.0.self_attn.rotary_emb.inv_freq";
  with_buffer.push_back(inv);
  r = check_checkpoint(c, with_buffer);
  CHECK(r.ok());
  CHECK_EQ(r.warnings.size(), std::size_t{1});

  auto int_dtype = good;
  int_dtype[0].dtype = DType::I32;
  r = check_checkpoint(c, int_dtype);
  CHECK(!r.ok() && r.errors.at(0).find("non-float dtype") != std::string::npos);
}

TEST(config_load_from_file) {
  const ef_test::TempFile f(kTinyLlamaConfig, ".json");
  CHECK_EQ(load_model_config(f.path()).hidden_size, 2048);
  CHECK_THROWS(load_model_config("/no/such/config.json"), "cannot open");
  const ef_test::TempFile bad("{\"model_type\": \"llama\",}", ".json");
  CHECK_THROWS(load_model_config(bad.path()), "trailing comma");
}
