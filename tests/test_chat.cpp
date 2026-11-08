// EdgeForge - Stage 8 tests: when a chat reply ends (ReplyGuard, ChatSession stop reasons).
//
// The ChatSession tests use a scripted tiny model: every layer matrix is zero, so the residual
// stream is just the token embedding, and the embedding / lm_head are set so that the greedy next
// token is a fixed function of the current token. That lets a test write the exact reply the
// "model" produces: a normal answer ending in EOS, a reply that starts the next turn itself, a
// loop, or a reply where EOS is a close second choice.
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ef/chat.h"
#include "ef/error.h"
#include "ef/model.h"
#include "ef/tokenizer.h"
#include "test_framework.h"
#include "test_util.h"
#include "tiny_model.h"

using namespace ef;

namespace {

using Ids = std::vector<std::int32_t>;
using ef_test::Tiny;
using ef_test::TinyDir;

Tokenizer mini() { return Tokenizer::from_json(ef_test::mini_tokenizer_json(), ef_test::mini_tokenizer_config()); }

// Mini-tokenizer ids: letters a b c h i have their own pieces, everything else is a byte token.
constexpr std::int32_t kA = 265, kB = 266, kC = 267, kH = 269, kI = 270, kEos = 2;
constexpr std::int32_t byte_id(unsigned char c) { return 3 + c; }
constexpr std::int32_t kNewline = byte_id('\n');

struct Fed {
  std::string shown;
  std::size_t pushed = 0;
};

// Feed ids until the guard stops; returns what was shown and how many tokens it took.
Fed feed(ReplyGuard& g, const Ids& ids) {
  Fed f;
  for (std::int32_t id : ids) {
    f.shown += g.push(id);
    ++f.pushed;
    if (g.reason() != StopReason::None) return f;
  }
  f.shown += g.finish();
  return f;
}

Ids text_ids(const Tokenizer& t, const std::string& s) { return t.encode(s, /*add_bos=*/false, /*parse_special=*/false); }

// A tiny model whose greedy next token depends only on the current token: next[t].
// eos_second: tokens after which EOS is a close second choice.
struct ScriptedModel {
  Tiny t;
  std::unique_ptr<TinyDir> dir;
  ScriptedModel(const std::map<std::int32_t, std::int32_t>& next, const std::vector<std::int32_t>& eos_second = {}) {
    t.V = 272;  // the mini tokenizer
    std::map<std::int32_t, std::size_t> dim;
    for (const auto& [src, dst] : next) dim.emplace(src, dim.size());
    if (dim.size() >= t.D) throw Error("ScriptedModel: too many source tokens");
    const std::size_t D = t.D;
    auto tweak = [=](const std::string& name, std::uint64_t i, float) -> float {
      if (name.find("norm") != std::string::npos) return 1.0f;
      const auto row = static_cast<std::int32_t>(i / D);
      const auto col = static_cast<std::size_t>(i % D);
      if (name == "model.embed_tokens.weight") {
        const auto it = dim.find(row);
        return col == (it == dim.end() ? D - 1 : it->second) ? 1.0f : 0.0f;
      }
      if (name == "lm_head.weight") {
        for (const auto& [src, dst] : next) {
          if (col != dim.at(src)) continue;
          if (row == dst) return 10.0f;
          if (row == kEos && std::find(eos_second.begin(), eos_second.end(), src) != eos_second.end()) return 9.9f;
        }
        return 0.0f;
      }
      return 0.0f;  // every layer matrix: the residual stream stays the embedding
    };
    dir = std::make_unique<TinyDir>(t, 1, false, tweak);
  }
};

ChatOptions small_opts() {
  ChatOptions o;
  o.max_seq = 256;
  o.max_new = 64;
  return o;
}

}  // namespace

TEST(chat_guard_role_marker_split_across_tokens_is_never_shown) {
  const Tokenizer t = mini();
  const Ids ids = text_ids(t, "Fine, thanks!\n<|user|>\nhi");
  ReplyGuard g(t, StopConfig{});
  const Fed f = feed(g, ids);
  CHECK(g.reason() == StopReason::StopString);
  CHECK_EQ(g.detail(), std::string("<|user|>"));
  CHECK_EQ(g.reply(), std::string("Fine, thanks!"));
  CHECK_EQ(f.shown, std::string("Fine, thanks!\n"));  // nothing of "<|user|>" leaked while streaming
  CHECK(f.pushed < ids.size());                       // stopped at the marker, "hi" never consumed
  // The other markers, and </s> written as text
  for (const std::string m : {"<|assistant|>", "<|system|>", "</s>"}) {
    ReplyGuard g2(t, StopConfig{});
    const Fed f2 = feed(g2, text_ids(t, "ab" + m + "ca"));
    CHECK(g2.reason() == StopReason::StopString);
    CHECK_EQ(g2.detail(), m);
    CHECK_EQ(g2.reply(), std::string("ab"));
    CHECK_EQ(f2.shown, std::string("ab"));
  }
}

TEST(chat_guard_keeps_ordinary_text) {
  const Tokenizer t = mini();
  // Look-alikes that are not role markers, a held-back "<|" that turns out harmless, UTF-8.
  const std::string text =
      "Response: I'm fine.\nUser: hello\nAssistant: ok <b>x</b> <|us a <| b </ s> <|user| \xC3\xA9\xF0\x9F\x98\x80 end ";
  const Ids ids = text_ids(t, text);
  ReplyGuard g(t, StopConfig{});
  const Fed f = feed(g, ids);
  CHECK(g.reason() == StopReason::None);
  CHECK_EQ(f.pushed, ids.size());
  CHECK_EQ(g.reply(), t.decode(ids, true));  // EOS / limit: the decoded reply, unchanged (not trimmed)
  CHECK_EQ(f.shown, g.reply());              // streamed text == final text
  CHECK_EQ(g.reply(), text);
  // A reply that ends while a marker prefix is held back: released at the end.
  ReplyGuard g2(t, StopConfig{});
  const Fed f2 = feed(g2, text_ids(t, "use <|"));
  CHECK_EQ(f2.shown, std::string("use <|"));
  // Stops off: markers are just text.
  ReplyGuard g3(t, StopConfig::none());
  const Fed f3 = feed(g3, text_ids(t, "ab\n<|user|>\nhi"));
  CHECK(g3.reason() == StopReason::None);
  CHECK_EQ(f3.shown, std::string("ab\n<|user|>\nhi"));
}

TEST(chat_guard_repetition) {
  const Tokenizer t = mini();
  StopConfig c;  // span 12, count 3
  // A 3-token loop: the 12-token span recurs every 3 tokens -> third occurrence after 12 + 2*3 tokens.
  Ids loop;
  for (int k = 0; k < 20; ++k) loop.insert(loop.end(), {kA, kB, kC});
  ReplyGuard g(t, c);
  const Fed f = feed(g, loop);
  CHECK(g.reason() == StopReason::Repetition);
  CHECK_EQ(f.pushed, std::size_t{18});
  CHECK_EQ(g.reply(), std::string("abc"));  // cut where the repeat starts
  // An 8-token "sentence" repeated: fires on its third occurrence, keeps the first.
  const Ids sent = {kH, kI, byte_id(','), byte_id(' '), kA, kB, byte_id('.'), kNewline};
  Ids lines;
  for (int k = 0; k < 6; ++k) lines.insert(lines.end(), sent.begin(), sent.end());
  ReplyGuard g2(t, c);
  const Fed f2 = feed(g2, lines);
  CHECK(g2.reason() == StopReason::Repetition);
  CHECK_EQ(f2.pushed, std::size_t{12 + 2 * 8});
  CHECK_EQ(g2.reply(), std::string("hi, ab."));
  // Twice is not a loop; nor is varied text with shared phrases.
  Ids twice(lines.begin(), lines.begin() + 16);
  twice.insert(twice.end(), {kC, kC, byte_id('!')});
  ReplyGuard g3(t, c);
  feed(g3, twice);
  CHECK(g3.reason() == StopReason::None);
  const Ids varied = text_ids(t, "1. a cab: abc\n2. a bac: bca\n3. a acb: cab\n4. a cba: bac\n5. a bca: acb\n");
  ReplyGuard g4(t, c);
  feed(g4, varied);
  CHECK(g4.reason() == StopReason::None);
  // Configurable: count 0 = off; a shorter span catches it earlier.
  c.repeat_count = 0;
  ReplyGuard g5(t, c);
  feed(g5, loop);
  CHECK(g5.reason() == StopReason::None);
  c.repeat_count = 3;
  c.repeat_span = 3;
  ReplyGuard g6(t, c);
  const Fed f6 = feed(g6, loop);
  CHECK(g6.reason() == StopReason::Repetition);
  CHECK_EQ(f6.pushed, std::size_t{9});
}

TEST(chat_eos_ends_reply_and_max_new_is_last_bound) {
  // "\n" (end of the prompt) -> a -> b -> EOS
  const ScriptedModel sm({{kNewline, kA}, {kA, kB}, {kB, kEos}});
  const Model m = Model::load(sm.dir->path);
  const Tokenizer tok = mini();
  ChatOptions o = small_opts();
  o.system_prompt = "h";
  ChatSession chat(m, tok, o);
  TurnStats s;
  std::string streamed;
  CHECK_EQ(chat.reply("hi", [&](const std::string& x) { streamed += x; }, &s), std::string("ab"));
  CHECK(s.reason == StopReason::Eos && s.hit_eos && !s.hit_limit && !s.cut);
  CHECK_EQ(streamed, std::string("ab"));
  CHECK(s.reply_ids == (Ids{kA, kB}));
  CHECK(s.prompt_ids == tok.encode(apply_chat_template({{"system", "h"}, {"user", "hi"}}, true), true, true));
  CHECK(s.eos_p_max > 0.9);
  CHECK_EQ(s.eos_p_max_step, std::size_t{2});
  // A loop that never ends: with repetition off, max-new is what stops it.
  const ScriptedModel loop({{kNewline, kA}, {kA, kB}, {kB, kC}, {kC, kA}});
  const Model ml = Model::load(loop.dir->path);
  o.stop.repeat_count = 0;
  o.max_new = 20;
  ChatSession c2(ml, tok, o);
  c2.reply("hi", {}, &s);
  CHECK(s.reason == StopReason::MaxNew && s.hit_limit);
  CHECK_EQ(s.generated, std::size_t{20});
  // The context, not --max-new, runs out first.
  o.max_seq = s.prompt_tokens + 5;
  o.max_new = 64;
  ChatSession c3(ml, tok, o);
  c3.reply("hi", {}, &s);
  CHECK(s.reason == StopReason::Context && s.hit_limit);
  CHECK_EQ(s.generated, std::size_t{5});
}

TEST(chat_reply_cut_at_role_marker_and_multi_turn_continues) {
  // The model answers "ab", then starts the next turn itself: "</s>" written as text, then "c...".
  const ScriptedModel sm({{kNewline, kA},
                          {kA, kB},
                          {kB, byte_id('<')},
                          {byte_id('<'), byte_id('/')},
                          {byte_id('/'), byte_id('s')},
                          {byte_id('s'), byte_id('>')},
                          {byte_id('>'), kC},
                          {kC, kC}});
  const Model m = Model::load(sm.dir->path);
  const Tokenizer tok = mini();
  ChatOptions o = small_opts();
  ChatSession chat(m, tok, o);
  TurnStats s1, s2;
  std::string streamed;
  CHECK_EQ(chat.reply("hi", [&](const std::string& x) { streamed += x; }, &s1), std::string("ab"));
  CHECK(s1.reason == StopReason::StopString && s1.cut);
  CHECK_EQ(s1.stop_detail, std::string("</s>"));
  CHECK_EQ(streamed, std::string("ab"));  // "</s" was held back, never printed
  // "</s>" is 4 byte tokens here; the stream decoder releases a byte run when the next normal token
  // ("c") ends it, so the stop fires one token later (TinyLlama has normal pieces for "</", "s", ">").
  CHECK_EQ(s1.generated, std::size_t{7});
  CHECK_EQ(chat.messages().back().content, std::string("ab"));
  // Turn 2 sees the cut reply in its history and reuses the cached prefix.
  CHECK_EQ(chat.reply("ab", {}, &s2), std::string("ab"));
  CHECK(s2.reused_tokens >= s1.prompt_tokens);
  CHECK(s2.prompt_ids ==
        tok.encode(apply_chat_template({{"user", "hi"}, {"assistant", "ab"}, {"user", "ab"}}, true), true, true));
  // Raw behaviour (stops off): the model runs on to max-new.
  o.stop = StopConfig::none();
  o.max_new = 12;
  ChatSession raw(m, tok, o);
  TurnStats s3;
  CHECK_EQ(raw.reply("hi", {}, &s3), std::string("ab</s>cccccc"));
  CHECK(s3.reason == StopReason::MaxNew);
  // set_history replays a given conversation (verify-chat).
  raw.set_history({{"system", "h"}, {"user", "x"}, {"assistant", "ab"}});
  raw.reply("hi", {}, &s3);
  CHECK(s3.prompt_ids ==
        tok.encode(apply_chat_template({{"system", "h"}, {"user", "x"}, {"assistant", "ab"}, {"user", "hi"}}, true), true,
                   true));
  CHECK_THROWS(raw.set_history({{"tool", "x"}}), "unknown role");
}

TEST(chat_repetition_stop_in_session_and_eos_min_p) {
  const ScriptedModel loop({{kNewline, kA}, {kA, kB}, {kB, kC}, {kC, kA}});
  const Model m = Model::load(loop.dir->path);
  const Tokenizer tok = mini();
  ChatOptions o = small_opts();
  ChatSession chat(m, tok, o);
  TurnStats s;
  CHECK_EQ(chat.reply("hi", {}, &s), std::string("abc"));
  CHECK(s.reason == StopReason::Repetition && s.cut);
  CHECK_EQ(s.generated, std::size_t{18});
  // EOS as a close second after "b": greedy continues; --eos-min-p ends the turn there.
  const ScriptedModel close({{kNewline, kA}, {kA, kB}, {kB, kC}, {kC, kC}}, {kB});
  const Model mc = Model::load(close.dir->path);
  o.stop.repeat_count = 0;
  o.max_new = 8;
  ChatSession greedy(mc, tok, o);
  greedy.reply("hi", {}, &s);
  CHECK(s.reason == StopReason::MaxNew);
  CHECK(s.eos_p_max > 0.3 && s.eos_p_max < 0.5);
  CHECK_EQ(s.eos_p_max_step, std::size_t{2});  // the logits after "b"
  o.stop.eos_min_p = 0.3f;
  ChatSession early(mc, tok, o);
  CHECK_EQ(early.reply("hi", {}, &s), std::string("ab"));
  CHECK(s.reason == StopReason::EosMinP && s.hit_eos);
}
