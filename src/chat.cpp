// EdgeForge - Stage 4 chat session.
#include "ef/chat.h"

#include <algorithm>
#include <chrono>

#include "ef/error.h"

namespace ef {

namespace {
const Model& checked(const Model& model, const Tokenizer& tok, const ChatOptions& opt) {
  if (tok.vocab_size() > static_cast<std::size_t>(model.config().vocab_size)) {
    throw Error("tokenizer has " + std::to_string(tok.vocab_size()) + " tokens but the model only " +
                std::to_string(model.config().vocab_size));
  }
  if (opt.max_new == 0) throw Error("chat: max_new must be > 0");
  return model;
}
}  // namespace

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
  StreamDecoder stream(tok_);
  for (std::size_t step = 0; step < max_new; ++step) {
    const std::int32_t tok = sampler_.sample(logits, V, history);
    if (tok == tok_.eos_id()) {
      st.hit_eos = true;
      break;
    }
    generated.push_back(tok);
    history.push_back(tok);
    const std::string delta = stream.push(tok);
    if (on_text && !delta.empty()) on_text(delta);
    if (step + 1 == max_new) {
      st.hit_limit = true;
      break;
    }
    logits = tr_.forward(tok, cached_.size());
    cached_.push_back(tok);
  }
  const std::string tail = stream.flush();
  if (on_text && !tail.empty()) on_text(tail);
  st.decode_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
  st.generated = generated.size();

  std::string text = tok_.decode(generated, /*skip_special=*/true);
  messages_.push_back({"assistant", text});
  if (stats != nullptr) *stats = st;
  return text;
}

}  // namespace ef
