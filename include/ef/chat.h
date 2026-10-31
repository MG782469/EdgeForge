// EdgeForge - Stage 4: multi-turn chat on top of the FP32 engine.
// Each turn re-renders the whole conversation with the chat template, tokenizes it,
// and only feeds the tokens after the longest prefix already in the KV cache, so
// earlier turns are never recomputed.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "ef/model.h"
#include "ef/sampler.h"
#include "ef/tokenizer.h"

namespace ef {

struct ChatOptions {
  std::size_t max_seq = 2048;  // KV-cache capacity (context length)
  std::size_t max_new = 256;   // reply length limit
  SamplerConfig sampler;       // default: greedy
  std::string system_prompt;   // optional
};

struct TurnStats {
  std::size_t prompt_tokens = 0;   // whole conversation, tokenized
  std::size_t reused_tokens = 0;   // already in the KV cache
  std::size_t prefill_tokens = 0;  // actually computed this turn
  std::size_t generated = 0;       // reply tokens (EOS excluded)
  std::size_t dropped_messages = 0;
  double prefill_s = 0.0, decode_s = 0.0;
  bool hit_eos = false, hit_limit = false;
};

class ChatSession {
 public:
  ChatSession(const Model& model, const Tokenizer& tok, const ChatOptions& opt);

  // Generates the assistant reply; on_text receives the text as it streams.
  std::string reply(const std::string& user_text, const std::function<void(const std::string&)>& on_text = {},
                    TurnStats* stats = nullptr);
  void reset();
  const std::vector<ChatMessage>& messages() const noexcept { return messages_; }
  std::size_t cached_tokens() const noexcept { return cached_.size(); }

 private:
  std::vector<std::int32_t> render() const;

  const Model& m_;
  const Tokenizer& tok_;
  ChatOptions opt_;
  Transformer tr_;
  Sampler sampler_;
  std::vector<ChatMessage> messages_;
  std::vector<std::int32_t> cached_;  // tokens whose K/V are in the cache, positions 0..n-1
};

}  // namespace ef
