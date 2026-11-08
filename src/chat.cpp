// EdgeForge - Stage 4 chat session.
#include "ef/chat.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include "ef/error.h"
#include "ef/format.h"

namespace ef {

const char* stop_reason_name(StopReason r) noexcept {
  switch (r) {
    case StopReason::None: return "none";
    case StopReason::Eos: return "EOS";
    case StopReason::EosMinP: return "EOS probability";
    case StopReason::StopString: return "role marker";
    case StopReason::Repetition: return "repetition";
    case StopReason::MaxNew: return "max-new limit";
    case StopReason::Context: return "context full";
  }
  return "?";
}

namespace {

std::string rtrim(std::string s) {
  while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\t' || s.back() == '\r')) s.pop_back();
  return s;
}

// P(token) under softmax(logits), in double precision.
double prob_of(const float* logits, std::size_t n, std::int32_t id) {
  if (id < 0 || static_cast<std::size_t>(id) >= n) return 0.0;
  double m = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < n; ++i) m = std::max(m, static_cast<double>(logits[i]));
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) s += std::exp(static_cast<double>(logits[i]) - m);
  return std::exp(static_cast<double>(logits[static_cast<std::size_t>(id)]) - m) / s;
}

const Model& checked(const Model& model, const Tokenizer& tok, const ChatOptions& opt) {
  if (tok.vocab_size() > static_cast<std::size_t>(model.config().vocab_size)) {
    throw Error("tokenizer has " + std::to_string(tok.vocab_size()) + " tokens but the model only " +
                std::to_string(model.config().vocab_size));
  }
  if (opt.max_new == 0) throw Error("chat: max_new must be > 0");
  return model;
}
}  // namespace

// ---- ReplyGuard ---------------------------------------------------------------------------

ReplyGuard::ReplyGuard(const Tokenizer& tok, StopConfig cfg) : tok_(tok), cfg_(std::move(cfg)), stream_(tok) {
  cfg_.stop_strings.erase(std::remove(cfg_.stop_strings.begin(), cfg_.stop_strings.end(), std::string()),
                          cfg_.stop_strings.end());
  for (const auto& s : cfg_.stop_strings) max_stop_len_ = std::max(max_stop_len_, s.size());
}

// Earliest stop string starting at or after `from`.
std::size_t ReplyGuard::find_stop(std::size_t from, std::string* which) const {
  std::size_t best = std::string::npos;
  for (const auto& s : cfg_.stop_strings) {
    const std::size_t p = text_.find(s, from);
    if (p < best) {
      best = p;
      *which = s;
    }
  }
  return best;
}

void ReplyGuard::stop_at_text(std::size_t pos, const std::string& which) {
  reason_ = StopReason::StopString;
  detail_ = which;
  reply_ = rtrim(text_.substr(0, pos));
}

// Text from shown_ on that is safe to show. Unless final, hold back the longest tail that is a
// proper prefix of a stop string ("<|us" might become "<|user|>").
std::string ReplyGuard::release(bool final) {
  std::size_t end = text_.size();
  if (!final) {
    std::size_t hold = 0;
    for (const auto& s : cfg_.stop_strings) {
      for (std::size_t k = std::min(s.size() - 1, end - shown_); k > hold; --k) {
        if (text_.compare(end - k, k, s, 0, k) == 0) {
          hold = k;
          break;
        }
      }
    }
    end -= hold;
  }
  std::string out = text_.substr(shown_, end - shown_);
  shown_ = end;
  return out;
}

std::string ReplyGuard::push(std::int32_t id) {
  if (reason_ != StopReason::None) return {};
  ids_.push_back(id);
  const std::size_t old = text_.size();
  text_ += stream_.push(id);
  if (max_stop_len_ > 0) {
    std::string which;
    const std::size_t p = find_stop(old >= max_stop_len_ ? old - (max_stop_len_ - 1) : 0, &which);
    if (p != std::string::npos) {
      std::string out = p > shown_ ? text_.substr(shown_, p - shown_) : std::string();
      shown_ = std::max(shown_, p);
      stop_at_text(p, which);
      return out;
    }
  }
  if (cfg_.repeat_count >= 2 && cfg_.repeat_span > 0 && ids_.size() >= cfg_.repeat_span) {
    const std::size_t start = ids_.size() - cfg_.repeat_span;
    auto& seen = spans_[std::vector<std::int32_t>(ids_.begin() + static_cast<std::ptrdiff_t>(start), ids_.end())];
    seen.push_back(start);
    if (seen.size() >= cfg_.repeat_count) {
      reason_ = StopReason::Repetition;
      detail_ = strf("a %zu-token span occurred %zu times", cfg_.repeat_span, seen.size());
      const std::vector<std::int32_t> keep(ids_.begin(), ids_.begin() + static_cast<std::ptrdiff_t>(seen[1]));
      reply_ = rtrim(tok_.decode(keep, /*skip_special=*/true));
      return {};
    }
  }
  return release(false);
}

std::string ReplyGuard::finish() {
  if (reason_ != StopReason::None) return {};
  const std::size_t old = text_.size();
  text_ += stream_.flush();
  if (max_stop_len_ > 0 && text_.size() > old) {
    std::string which;
    const std::size_t p = find_stop(old >= max_stop_len_ ? old - (max_stop_len_ - 1) : 0, &which);
    if (p != std::string::npos) {
      std::string out = p > shown_ ? text_.substr(shown_, p - shown_) : std::string();
      stop_at_text(p, which);
      return out;
    }
  }
  reply_ = text_;  // ended by EOS or a limit: the decoded reply, unchanged
  return release(true);
}

// ---- ChatSession --------------------------------------------------------------------------

ChatSession::ChatSession(const Model& model, const Tokenizer& tok, const ChatOptions& opt)
    : m_(checked(model, tok, opt)), tok_(tok), opt_(opt), tr_(model, opt.max_seq, opt.pool), sampler_(opt.sampler) {
  reset();
}

void ChatSession::reset() {
  messages_.clear();
  cached_.clear();
  tr_.reset();
  if (!opt_.system_prompt.empty()) messages_.push_back({"system", opt_.system_prompt});
}

void ChatSession::set_history(std::vector<ChatMessage> messages) {
  for (const auto& m : messages) {
    if (m.role != "system" && m.role != "user" && m.role != "assistant") throw Error("chat: unknown role '" + m.role + "'");
  }
  messages_ = std::move(messages);
}

std::vector<std::int32_t> ChatSession::render() const {
  return tok_.encode(apply_chat_template(messages_, true, tok_.eos_token()), /*add_bos=*/true, /*parse_special=*/true);
}

std::string ChatSession::reply(const std::string& user_text, const std::function<void(const std::string&)>& on_text,
                               TurnStats* stats) {
  TurnStats st;
  messages_.push_back({"user", user_text});
  std::vector<std::int32_t> ids = render();
  // Keep room for at least a short reply: drop the oldest exchanges (never the system prompt).
  const std::size_t reserve = std::min<std::size_t>(opt_.max_new, 64);
  while (ids.size() + reserve > opt_.max_seq) {
    const std::size_t first = (!messages_.empty() && messages_[0].role == "system") ? 1 : 0;
    if (messages_.size() - first <= 1) break;  // only the new user message is left
    messages_.erase(messages_.begin() + static_cast<std::ptrdiff_t>(first));
    ++st.dropped_messages;
    ids = render();
  }
  if (ids.size() >= opt_.max_seq) {
    messages_.pop_back();
    throw Error("message is too long for the " + std::to_string(opt_.max_seq) + "-token context");
  }
  st.prompt_tokens = ids.size();

  // Reuse the cached prefix; always recompute at least the last prompt token (we need its logits).
  std::size_t lcp = 0;
  while (lcp < cached_.size() && lcp < ids.size() && cached_[lcp] == ids[lcp]) ++lcp;
  if (lcp == ids.size()) --lcp;
  st.reused_tokens = lcp;
  st.prefill_tokens = ids.size() - lcp;

  const auto V = static_cast<std::size_t>(m_.config().vocab_size);
  const auto t0 = std::chrono::steady_clock::now();
  const float* logits = nullptr;
  for (std::size_t p = lcp; p < ids.size(); ++p) logits = tr_.forward(ids[p], p, p + 1 == ids.size());
  cached_ = ids;
  const auto t1 = std::chrono::steady_clock::now();
  st.prefill_s = std::chrono::duration<double>(t1 - t0).count();

  const std::size_t max_new = std::min(opt_.max_new, opt_.max_seq - ids.size());
  std::vector<std::int32_t> history = ids, generated;
  ReplyGuard guard(tok_, opt_.stop);
  auto emit = [&](const std::string& t) {
    if (on_text && !t.empty()) on_text(t);
  };
  for (std::size_t step = 0; step < max_new; ++step) {
    std::int32_t tok = sampler_.sample(logits, V, history);
    const double p_eos = prob_of(logits, V, tok_.eos_id());
    st.eos_p_steps.push_back(p_eos);
    if (p_eos > st.eos_p_max) {
      st.eos_p_max = p_eos;
      st.eos_p_max_step = step;
    }
    if (tok == tok_.eos_id()) {
      st.reason = StopReason::Eos;
      break;
    }
    if (opt_.stop.eos_min_p > 0.0f && p_eos >= static_cast<double>(opt_.stop.eos_min_p)) {
      st.reason = StopReason::EosMinP;
      break;
    }
    generated.push_back(tok);
    history.push_back(tok);
    emit(guard.push(tok));
    if (guard.reason() != StopReason::None) {
      st.reason = guard.reason();
      st.stop_detail = guard.detail();
      break;
    }
    if (step + 1 == max_new) {
      st.reason = max_new < opt_.max_new ? StopReason::Context : StopReason::MaxNew;
      break;
    }
    logits = tr_.forward(tok, cached_.size());
    cached_.push_back(tok);
  }
  emit(guard.finish());
  st.hit_eos = st.reason == StopReason::Eos || st.reason == StopReason::EosMinP;
  st.hit_limit = st.reason == StopReason::MaxNew || st.reason == StopReason::Context;
  st.decode_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
  st.generated = generated.size();

  std::string text = guard.reply();
  st.cut = st.reason == StopReason::StopString || st.reason == StopReason::Repetition;
  st.prompt_ids = ids;
  st.reply_ids = std::move(generated);
  messages_.push_back({"assistant", text});
  if (stats != nullptr) *stats = st;
  return text;
}

}  // namespace ef
