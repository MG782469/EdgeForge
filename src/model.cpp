// EdgeForge - Stage 3: FP32 reference forward pass with KV cache.
#include "ef/model.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>

#include "ef/error.h"

namespace ef {

namespace {

std::vector<float> read_f32_tensor(const SafetensorsFile& st, const std::string& name) {
  const TensorInfo& t = st.at(name);
  std::vector<float> v(static_cast<std::size_t>(t.numel));
  st.read_f32(t, 0, t.numel, v.data());
  return v;
}

}  // namespace

Model Model::load(const std::filesystem::path& model_dir) {
  Model m;
  m.cfg_ = load_model_config(model_dir / "config.json");
  m.st_ = SafetensorsFile::open(model_dir / "model.safetensors");
  const CheckReport rep = check_checkpoint(m.cfg_, m.st_.tensors());
  if (!rep.ok()) throw Error("checkpoint does not match config.json: " + rep.errors.front());

  auto bf16 = [&](const std::string& name) {
    const TensorInfo& t = m.st_.at(name);
    m.bytes_per_token_ += t.nbytes();
    return m.st_.bf16_data(t);
  };
  m.embed_ = m.st_.bf16_data(m.st_.at("model.embed_tokens.weight"));  // one row per token: not streamed
  m.lm_head_ = m.cfg_.tie_word_embeddings ? m.embed_ : bf16("lm_head.weight");
  if (m.cfg_.tie_word_embeddings) m.bytes_per_token_ += m.st_.at("model.embed_tokens.weight").nbytes();
  m.final_norm_ = read_f32_tensor(m.st_, "model.norm.weight");
  m.bytes_per_token_ += m.st_.at("model.norm.weight").nbytes();

  m.layers_.resize(static_cast<std::size_t>(m.cfg_.num_hidden_layers));
  for (std::size_t l = 0; l < m.layers_.size(); ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    LayerWeights& w = m.layers_[l];
    w.wq = bf16(p + "self_attn.q_proj.weight");
    w.wk = bf16(p + "self_attn.k_proj.weight");
    w.wv = bf16(p + "self_attn.v_proj.weight");
    w.wo = bf16(p + "self_attn.o_proj.weight");
    w.w_gate = bf16(p + "mlp.gate_proj.weight");
    w.w_up = bf16(p + "mlp.up_proj.weight");
    w.w_down = bf16(p + "mlp.down_proj.weight");
    w.attn_norm = read_f32_tensor(m.st_, p + "input_layernorm.weight");
    w.ffn_norm = read_f32_tensor(m.st_, p + "post_attention_layernorm.weight");
    m.bytes_per_token_ += m.st_.at(p + "input_layernorm.weight").nbytes();
    m.bytes_per_token_ += m.st_.at(p + "post_attention_layernorm.weight").nbytes();
  }
  return m;
}

Transformer::Transformer(const Model& model, std::size_t max_seq) : m_(model) {
  const ModelConfig& c = model.config();
  D_ = static_cast<std::size_t>(c.hidden_size);
  F_ = static_cast<std::size_t>(c.intermediate_size);
  H_ = static_cast<std::size_t>(c.num_attention_heads);
  KV_ = static_cast<std::size_t>(c.num_key_value_heads);
  hd_ = static_cast<std::size_t>(c.head_dim);
  kvd_ = KV_ * hd_;
  V_ = static_cast<std::size_t>(c.vocab_size);
  L_ = model.n_layers();
  eps_ = static_cast<float>(c.rms_norm_eps);
  if (max_seq == 0) throw Error("Transformer: max_seq must be > 0");
  if (max_seq > static_cast<std::size_t>(c.max_position_embeddings)) {
    throw Error("Transformer: max_seq " + std::to_string(max_seq) + " exceeds max_position_embeddings " +
                std::to_string(c.max_position_embeddings));
  }
  max_seq_ = max_seq;
  rope_ = kern::RopeTable(max_seq_, hd_, static_cast<float>(c.rope_theta));
  k_cache_.assign(L_ * max_seq_ * kvd_, 0.0f);
  v_cache_.assign(L_ * max_seq_ * kvd_, 0.0f);
  x_.resize(D_);
  h_.resize(D_);
  q_.resize(D_);
  ctx_.resize(D_);
  o_.resize(D_);
  gate_.resize(F_);
  up_.resize(F_);
  scores_.resize(max_seq_);
  logits_.resize(V_);
}

const float* Transformer::forward(std::int32_t token, std::size_t pos, bool compute_logits, ForwardTrace* trace) {
  if (token < 0 || static_cast<std::size_t>(token) >= V_) {
    throw Error("forward: token id " + std::to_string(token) + " is outside [0, " + std::to_string(V_) + ")");
  }
  if (pos >= max_seq_) {
    throw Error("forward: position " + std::to_string(pos) + " exceeds the KV-cache capacity " +
                std::to_string(max_seq_));
  }
  if (pos > cached_) {
    throw Error("forward: position " + std::to_string(pos) + " skips ahead of the " + std::to_string(cached_) +
                " cached tokens");
  }
  if (trace != nullptr && trace->capture) {
    trace->layer_out.resize(L_ * D_);
    trace->final_norm.resize(D_);
  }

  float* x = x_.data();
  float* h = h_.data();
  kern::embedding_bf16(x, m_.embed_tokens(), static_cast<std::size_t>(token), D_);
  const float* cos_row = rope_.cos_row(pos);
  const float* sin_row = rope_.sin_row(pos);

  for (std::size_t l = 0; l < L_; ++l) {
    const LayerWeights& w = m_.layer(l);
    float* k_layer = k_cache_.data() + l * max_seq_ * kvd_;
    float* v_layer = v_cache_.data() + l * max_seq_ * kvd_;
    float* k_pos = k_layer + pos * kvd_;
    float* v_pos = v_layer + pos * kvd_;

    // ---- attention block
    kern::rmsnorm(h, x, w.attn_norm.data(), D_, eps_);
    kern::matvec_bf16(q_.data(), w.wq, h, D_, D_);
    kern::matvec_bf16(k_pos, w.wk, h, kvd_, D_);
    kern::matvec_bf16(v_pos, w.wv, h, kvd_, D_);
    kern::rope_apply(q_.data(), H_, hd_, cos_row, sin_row);
    kern::rope_apply(k_pos, KV_, hd_, cos_row, sin_row);
    kern::attention(ctx_.data(), q_.data(), k_layer, v_layer, pos + 1, H_, KV_, hd_, scores_.data());
    kern::matvec_bf16(o_.data(), w.wo, ctx_.data(), D_, D_);
    kern::add_inplace(x, o_.data(), D_);

    // ---- MLP block (SwiGLU)
    kern::rmsnorm(h, x, w.ffn_norm.data(), D_, eps_);
    kern::matvec_bf16(gate_.data(), w.w_gate, h, F_, D_);
    kern::matvec_bf16(up_.data(), w.w_up, h, F_, D_);
    kern::swiglu(gate_.data(), gate_.data(), up_.data(), F_);
    kern::matvec_bf16(o_.data(), w.w_down, gate_.data(), D_, F_);
    kern::add_inplace(x, o_.data(), D_);

    if (trace != nullptr && trace->capture) std::memcpy(trace->layer_out.data() + l * D_, x, D_ * sizeof(float));
  }
  cached_ = pos + 1;  // positions > pos (if any) are now stale and logically dropped

  kern::rmsnorm(h, x, m_.final_norm().data(), D_, eps_);
  if (trace != nullptr && trace->capture) std::memcpy(trace->final_norm.data(), h, D_ * sizeof(float));
  if (!compute_logits) return nullptr;
  kern::matvec_bf16(logits_.data(), m_.lm_head(), h, V_, D_);
  return logits_.data();
}

GreedyResult generate_greedy(Transformer& t, const std::vector<std::int32_t>& prompt, std::size_t max_new,
                             std::int32_t eos_id, std::size_t keep_step_logits) {
  if (prompt.empty()) throw Error("generate: empty prompt");
  if (prompt.size() + max_new > t.max_seq()) {
    throw Error("generate: prompt (" + std::to_string(prompt.size()) + ") + max_new (" + std::to_string(max_new) +
                ") exceeds the KV-cache capacity " + std::to_string(t.max_seq()));
  }
  const std::size_t V = static_cast<std::size_t>(t.model().config().vocab_size);
  GreedyResult r;
  t.reset();
  const auto t0 = std::chrono::steady_clock::now();
  const float* logits = nullptr;
  for (std::size_t p = 0; p < prompt.size(); ++p) logits = t.forward(prompt[p], p, p + 1 == prompt.size());
  const auto t1 = std::chrono::steady_clock::now();
  r.prefill_seconds = std::chrono::duration<double>(t1 - t0).count();
  std::size_t pos = prompt.size();
  for (std::size_t step = 0; step < max_new; ++step) {
    const auto tok = static_cast<std::int32_t>(kern::argmax(logits, V));
    r.tokens.push_back(tok);
    if (tok == eos_id || step + 1 == max_new) break;
    logits = t.forward(tok, pos++);
    ++r.decode_forwards;
    if (r.step_logits.size() < keep_step_logits) r.step_logits.emplace_back(logits, logits + V);
  }
  r.decode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
  return r;
}

}  // namespace ef
