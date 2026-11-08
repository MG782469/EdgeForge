// EdgeForge - Stage 4: multi-turn chat on top of the FP32 engine.
// Each turn re-renders the whole conversation with the chat template, tokenizes it,
// and only feeds the tokens after the longest prefix already in the KV cache, so
// earlier turns are never recomputed.
// Stage 8: when a reply ends. EOS ends it, as before. ReplyGuard adds two stops that only fire on
// clear signs that the model has left its turn: a chat-template role marker in the text (the model
// writing the next "<|user|>" turn itself), or a span of tokens repeated over and over. The
// max-new limit stays as the last safety bound. Ordinary text such as "Response:" never stops it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "ef/model.h"
#include "ef/sampler.h"
#include "ef/tokenizer.h"

namespace ef {

// Stage 8: system prompt of `edgeforge chat` and `verify-chat` unless --system / --no-system.
// It is only the content of TinyLlama's own <|system|> turn; the role format is unchanged.
inline constexpr const char* kDefaultSystemPrompt =
    "You are a helpful assistant. Give one short, direct reply to the user's message.";

struct StopConfig {
  // Text that can only mean "the model is writing another turn": the template's role markers and
  // the end-of-turn token written out as text. The reply is cut before the first one.
  std::vector<std::string> stop_strings = {"<|user|>", "<|assistant|>", "<|system|>", "</s>"};
  // Repetition: stop when any span of repeat_span tokens has occurred repeat_count times in the
  // reply (repeat_count < 2 = off). The reply is cut where the span's second occurrence starts.
  std::size_t repeat_count = 3, repeat_span = 12;
  // > 0: also end the turn when the model gives EOS at least this probability (off by default).
  float eos_min_p = 0.0f;

  static StopConfig none() { return {{}, 0, 0, 0.0f}; }  // EOS and max-new only (raw model behaviour)
};

enum class StopReason { None, Eos, EosMinP, StopString, Repetition, MaxNew, Context };
const char* stop_reason_name(StopReason r) noexcept;

// Watches the generated tokens of one reply. push() returns the text that is safe to show now:
// text that could still turn out to be the start of a stop string is held back until it cannot.
class ReplyGuard {
 public:
  ReplyGuard(const Tokenizer& tok, StopConfig cfg);
  std::string push(std::int32_t id);  // one generated token (never EOS)
  std::string finish();               // end of the reply: the held-back text (if no stop fired)
  StopReason reason() const noexcept { return reason_; }  // StopString / Repetition / None
  const std::string& detail() const noexcept { return detail_; }
  const std::string& reply() const noexcept { return reply_; }  // final text, after finish() or a stop
  std::size_t tokens() const noexcept { return ids_.size(); }

 private:
  std::size_t find_stop(std::size_t from, std::string* which) const;
  std::string release(bool final);
  void stop_at_text(std::size_t pos, const std::string& which);

  const Tokenizer& tok_;
  StopConfig cfg_;
  StreamDecoder stream_;
  std::vector<std::int32_t> ids_;
  std::string text_, reply_, detail_;
  std::size_t shown_ = 0, max_stop_len_ = 0;
  std::map<std::vector<std::int32_t>, std::vector<std::size_t>> spans_;  // span -> start positions
  StopReason reason_ = StopReason::None;
};

struct ChatOptions {
  std::size_t max_seq = 2048;  // KV-cache capacity (context length)
  std::size_t max_new = 256;   // reply length limit: the last safety bound
  SamplerConfig sampler;       // default: greedy
  std::string system_prompt;   // optional (the CLI passes kDefaultSystemPrompt)
  StopConfig stop;             // Stage 8: role markers + repetition
  ThreadPool* pool = nullptr;  // Stage 7: row-parallel matrix multiplies (nullptr = 1 thread)
};

struct TurnStats {
  std::size_t prompt_tokens = 0;   // whole conversation, tokenized
  std::size_t reused_tokens = 0;   // already in the KV cache
  std::size_t prefill_tokens = 0;  // actually computed this turn
  std::size_t generated = 0;       // reply tokens (EOS excluded)
  std::size_t dropped_messages = 0;
  double prefill_s = 0.0, decode_s = 0.0;
  bool hit_eos = false, hit_limit = false;
  // Stage 8
  StopReason reason = StopReason::None;
  std::string stop_detail;                          // which stop string / repeated span
  bool cut = false;                                 // reply text shorter than what was generated
  std::vector<std::int32_t> prompt_ids, reply_ids;  // reply_ids: every sampled token except EOS
  double eos_p_max = 0.0;                           // highest P(EOS) at any step of this reply
  std::size_t eos_p_max_step = 0;
  std::vector<double> eos_p_steps;                  // P(EOS) at every step (eos-study traces)
};

class ChatSession {
 public:
  ChatSession(const Model& model, const Tokenizer& tok, const ChatOptions& opt);

  // Generates the assistant reply; on_text receives the text as it streams.
  std::string reply(const std::string& user_text, const std::function<void(const std::string&)>& on_text = {},
                    TurnStats* stats = nullptr);
  void reset();
  // Replace the conversation so far (verify-chat replays a reference conversation); the KV cache
  // is kept and reused up to the first token that differs.
  void set_history(std::vector<ChatMessage> messages);
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
