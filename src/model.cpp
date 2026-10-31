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

  auto view = [&](const std::string& name) {
    const TensorInfo& t = m.st_.at(name);
    return WeightView{QType::BF16, reinterpret_cast<const std::uint8_t*>(m.st_.bf16_data(t)),
                      static_cast<std::size_t>(t.shape.at(0)), static_cast<std::size_t>(t.shape.at(1))};
  };
  auto streamed = [&](const std::string& name) {
    m.bytes_per_token_ += m.st_.at(name).nbytes();
    return view(name);
  };
  m.embed_ = view("model.embed_tokens.weight");  // one row per token: not streamed
  m.lm_head_ = m.cfg_.tie_word_embeddings ? m.embed_ : streamed("lm_head.weight");
  if (m.cfg_.tie_word_embeddings) m.bytes_per_token_ += m.st_.at("model.embed_tokens.weight").nbytes();
  m.final_norm_ = read_f32_tensor(m.st_, "model.norm.weight");
  m.bytes_per_token_ += m.st_.at("model.norm.weight").nbytes();

  m.layers_.resize(static_cast<std::size_t>(m.cfg_.num_hidden_layers));
  for (std::size_t l = 0; l < m.layers_.size(); ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    LayerWeights& w = m.layers_[l];
    w.wq = streamed(p + "self_attn.q_proj.weight");
    w.wk = streamed(p + "self_attn.k_proj.weight");
    w.wv = streamed(p + "self_attn.v_proj.weight");
    w.wo = streamed(p + "self_attn.o_proj.weight");
    w.w_gate = streamed(p + "mlp.gate_proj.weight");
    w.w_up = streamed(p + "mlp.up_proj.weight");
    w.w_down = streamed(p + "mlp.down_proj.weight");
    w.attn_norm = read_f32_tensor(m.st_, p + "input_layernorm.weight");
    w.ffn_norm = read_f32_tensor(m.st_, p + "post_attention_layernorm.weight");
    m.bytes_per_token_ += m.st_.at(p + "input_layernorm.weight").nbytes();
    m.bytes_per_token_ += m.st_.at(p + "post_attention_layernorm.weight").nbytes();
  }
  m.description_ = "BF16 safetensors";
  return m;
}

Model Model::load_efq(const std::filesystem::path& efq_file) {
  Model m;
  m.q_ = QuantFile::open(efq_file);
  m.cfg_ = m.q_.config();
  std::size_t n_q4 = 0, n_q8 = 0, n_other = 0;
  auto view = [&](const std::string& name) {
    const QTensor& t = m.q_.at(name);
    if (t.shape.size() != 2) throw Error("'" + name + "' must be a matrix");
    return WeightView{t.type, m.q_.data(t), static_cast<std::size_t>(t.rows()), static_cast<std::size_t>(t.cols())};
  };
  auto matrix = [&](const std::string& name) {
    const WeightView w = view(name);
    (w.type == QType::Q4_0 ? n_q4 : w.type == QType::Q8_0 ? n_q8 : n_other)++;
    return w;
  };
  auto vec = [&](const std::string& name) {
    const QTensor& t = m.q_.at(name);
    std::vector<float> v(static_cast<std::size_t>(t.cols()));
    m.q_.dequantize_row(t, 0, v.data());  // norms are F32: an exact copy
    return v;
  };
  m.embed_ = view("model.embed_tokens.weight");
  m.lm_head_ = m.cfg_.tie_word_embeddings ? m.embed_ : matrix("lm_head.weight");
  m.final_norm_ = vec("model.norm.weight");
  m.layers_.resize(static_cast<std::size_t>(m.cfg_.num_hidden_layers));
  for (std::size_t l = 0; l < m.layers_.size(); ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    LayerWeights& w = m.layers_[l];
    w.wq = matrix(p + "self_attn.q_proj.weight");
    w.wk = matrix(p + "self_attn.k_proj.weight");
    w.wv = matrix(p + "self_attn.v_proj.weight");
    w.wo = matrix(p + "self_attn.o_proj.weight");
    w.w_gate = matrix(p + "mlp.gate_proj.weight");
    w.w_up = matrix(p + "mlp.up_proj.weight");
    w.w_down = matrix(p + "mlp.down_proj.weight");
    w.attn_norm = vec(p + "input_layernorm.weight");
    w.ffn_norm = vec(p + "post_attention_layernorm.weight");
  }
  m.bytes_per_token_ = m.q_.bytes_per_token();
  const std::string& policy = m.q_.policy().name;
  m.description_ = "efq" + (policy.empty() ? std::string() : ", policy " + policy) + ": " + std::to_string(n_q4) +
                   " Q4_0 + " + std::to_string(n_q8) + " Q8_0" +
                   (n_other ? " + " + std::to_string(n_other) + " other" : std::string()) + " matrices, embedding " +
                   qtype_name(m.embed_.type);
  return m;
}

Model Model::dequantized_copy(const Model& src) {
  Model m;
  m.cfg_ = src.cfg_;
  m.final_norm_ = src.final_norm_;
  auto deq = [&](const WeightView& w) {
    m.owned_.emplace_back(w.rows * w.cols);  // inner buffers never move, so views stay valid
    std::vector<float>& buf = m.owned_.back();
    for (std::size_t r = 0; r < w.rows; ++r) kern::row_f32(buf.data() + r * w.cols, w, r);
    m.bytes_per_token_ += buf.size() * sizeof(float);
    return WeightView{QType::F32, reinterpret_cast<const std::uint8_t*>(buf.data()), w.rows, w.cols};
  };
  m.embed_ = deq(src.embed_);
  m.bytes_per_token_ = 0;  // the embedding contributes one row per token
  m.lm_head_ = src.cfg_.tie_word_embeddings ? m.embed_ : deq(src.lm_head_);
  if (src.cfg_.tie_word_embeddings) m.bytes_per_token_ += m.embed_.bytes();
  m.layers_.resize(src.layers_.size());
  for (std::size_t l = 0; l < src.layers_.size(); ++l) {
    const LayerWeights& s = src.layers_[l];
    LayerWeights& w = m.layers_[l];
    w.wq = deq(s.wq);
    w.wk = deq(s.wk);
    w.wv = deq(s.wv);
    w.wo = deq(s.wo);
    w.w_gate = deq(s.w_gate);
    w.w_up = deq(s.w_up);
    w.w_down = deq(s.w_down);
    w.attn_norm = s.attn_norm;
    w.ffn_norm = s.ffn_norm;
  }
  m.description_ = "F32 dequantized copy of " + src.description_;
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
  kern::row_f32(x, m_.embed_tokens(), static_cast<std::size_t>(token));
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
    kern::matvec(q_.data(), w.wq, h);
    kern::matvec(k_pos, w.wk, h);
    kern::matvec(v_pos, w.wv, h);
    kern::rope_apply(q_.data(), H_, hd_, cos_row, sin_row);
    kern::rope_apply(k_pos, KV_, hd_, cos_row, sin_row);
    kern::attention(ctx_.data(), q_.data(), k_layer, v_layer, pos + 1, H_, KV_, hd_, scores_.data());
    kern::matvec(o_.data(), w.wo, ctx_.data());
    kern::add_inplace(x, o_.data(), D_);

    // ---- MLP block (SwiGLU)
    kern::rmsnorm(h, x, w.ffn_norm.data(), D_, eps_);
    kern::matvec(gate_.data(), w.w_gate, h);
    kern::matvec(up_.data(), w.w_up, h);
    kern::swiglu(gate_.data(), gate_.data(), up_.data(), F_);
    kern::matvec(o_.data(), w.w_down, gate_.data());
    kern::add_inplace(x, o_.data(), D_);

    if (trace != nullptr && trace->capture) std::memcpy(trace->layer_out.data() + l * D_, x, D_ * sizeof(float));
  }
  cached_ = pos + 1;  // positions > pos (if any) are now stale and logically dropped

  kern::rmsnorm(h, x, m_.final_norm().data(), D_, eps_);
  if (trace != nullptr && trace->capture) std::memcpy(trace->final_norm.data(), h, D_ * sizeof(float));
  if (!compute_logits) return nullptr;
  kern::matvec(logits_.data(), m_.lm_head(), h);
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
