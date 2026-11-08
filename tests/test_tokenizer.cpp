// EdgeForge - Stage 4 tokenizer + sampler unit tests on a synthetic tokenizer.json.
// (The real TinyLlama tokenizer is checked against Hugging Face by `edgeforge verify-tokenizer`.)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "ef/sampler.h"
#include "ef/tokenizer.h"
#include "test_framework.h"
#include "test_util.h"

using namespace ef;
using ef_test::MiniTokOptions;
using ef_test::mini_tokenizer_config;
using ef_test::mini_tokenizer_json;
using Ids = std::vector<std::int32_t>;

namespace {
const std::string S = "\xE2\x96\x81";
const std::string FFFD = "\xEF\xBF\xBD";
constexpr std::int32_t B(int byte) { return 3 + byte; }  // id of <0xXX>
Tokenizer mini(const MiniTokOptions& o = {}) {
  return Tokenizer::from_json(mini_tokenizer_json(o), o.with_config ? mini_tokenizer_config() : std::string());
}
}  // namespace

TEST(tokenizer_bpe_merge_order) {
  const Tokenizer t = mini();
  CHECK_EQ(t.vocab_size(), std::size_t{272});
  CHECK_EQ(t.merge_count(), std::size_t{7});
  CHECK(t.prepend_scheme() == PrependScheme::First);
  CHECK(t.encode("ab") == (Ids{262}));       // S a b -> Sa b -> Sab
  CHECK(t.encode("abc") == (Ids{260, 263}));  // "b c" (rank 0) wins over "a b": Sa + bc
  CHECK(t.encode("aaa") == (Ids{260, 271}));  // S a a a -> Sa a a -> Sa aa (leftmost "a a")
  CHECK(t.encode(" ") == (Ids{264}));         // already starts with S: nothing prepended
  CHECK(t.encode("  ") == (Ids{259}));
  CHECK(t.encode("   ") == (Ids{268}));       // leftmost "S S" first, then "SS S"
  CHECK(t.encode("") == (Ids{}));
  CHECK(t.encode("ab", true) == (Ids{1, 262}));
}

TEST(tokenizer_special_tokens_and_prepend_first) {
  const Tokenizer t = mini();
  CHECK(t.encode("<s>ab") == (Ids{1, 261}));             // no S after a special token
  CHECK(t.encode("ab</s>ab") == (Ids{262, 2, 261}));
  CHECK(t.encode("</s></s>") == (Ids{2, 2}));
  const Ids raw = t.encode("</s>", false, /*parse_special=*/false);
  CHECK(!raw.empty() && std::find(raw.begin(), raw.end(), 2) == raw.end());
  CHECK(t.is_special(2) && !t.is_special(262));
  CHECK_EQ(t.bos_id(), 1);
  CHECK_EQ(t.eos_id(), 2);
  CHECK_EQ(t.eos_token(), std::string("</s>"));  // read from {"content": ...}
}

TEST(tokenizer_byte_fallback_and_unk) {
  const Tokenizer t = mini();
  CHECK(t.encode("\xC3\xA9") == (Ids{264, B(0xC3), B(0xA9)}));  // é not in vocab -> UTF-8 bytes
  CHECK(t.encode("a\nb") == (Ids{260, B(0x0A), 266}));
  CHECK(t.encode("a\xFF") == (Ids{260, B(0xFF)}));  // invalid UTF-8 byte still round-trips as a byte
  MiniTokOptions o;
  o.byte_fallback = false;
  const Tokenizer u = mini(o);
  CHECK(u.encode("xyz") == (Ids{264, 0}));  // unknown chars -> one fused <unk>
}

TEST(tokenizer_config_variants) {
  MiniTokOptions legacy;
  legacy.with_config = false;  // legacy normalizer alone: prepend to every segment
  const Tokenizer a = mini(legacy);
  CHECK(a.prepend_scheme() == PrependScheme::Always);
  CHECK(a.encode("<s>ab") == (Ids{1, 262}));
  CHECK(a.encode("ab</s>ab") == (Ids{262, 2, 262}));
  MiniTokOptions never;
  never.with_config = false;
  never.pre_tokenizer = "{\"type\":\"Metaspace\",\"replacement\":\"" + S + "\",\"prepend_scheme\":\"never\",\"split\":false}";
  const Tokenizer n = mini(never);
  CHECK(n.prepend_scheme() == PrependScheme::Never);
  CHECK(n.encode("ab") == (Ids{261}));
  MiniTokOptions pairs;
  pairs.pair_merges = true;  // ["a","b"] merge format (tokenizers >= 0.20)
  const Tokenizer p = mini(pairs);
  CHECK(p.encode("abc") == (Ids{260, 263}) && p.encode("aaa") == (Ids{260, 271}));
  CHECK_THROWS(Tokenizer::from_json("{\"model\":{\"type\":\"Unigram\",\"vocab\":[]}}"), "not supported");
  CHECK_THROWS(Tokenizer::from_json("{\"version\":\"1.0\"}"), "missing key \"model\"");
  CHECK_THROWS(Tokenizer::load("/no/such/dir"), "tokenizer.json not found");
}

TEST(tokenizer_decode) {
  const Tokenizer t = mini();
  CHECK_EQ(t.decode({264}), std::string(""));      // lone S -> " " -> leading space stripped
  CHECK_EQ(t.decode({259}), std::string(" "));     // only ONE leading space is stripped
  CHECK_EQ(t.decode({262, 263}), std::string("abbc"));
  CHECK_EQ(t.decode({260, 264, 262}), std::string("a  ab"));
  CHECK_EQ(t.decode({1, 261, 2}), std::string("<s>ab</s>"));
  CHECK_EQ(t.decode({1, 261, 2}, true), std::string("ab"));
  CHECK_EQ(t.decode({264, B(0xC3), B(0xA9)}), std::string("\xC3\xA9"));
  CHECK_EQ(t.decode({B(0xC3)}), FFFD);                        // invalid run -> one U+FFFD per byte
  CHECK_EQ(t.decode({B(0xE2), B(0x96)}), FFFD + FFFD);
  CHECK_EQ(t.decode(t.encode("a\nb  abc")), std::string("a\nb  abc"));
  CHECK_THROWS(t.decode({5000}), "outside the vocabulary");
}

TEST(tokenizer_stream_decoder_is_prefix_stable) {
  const Tokenizer t = mini();
  // emoji (4 bytes) + start of another 3-byte char + its last byte + normal tokens
  const Ids ids = {264, B(0xF0), B(0x9F), B(0x98), B(0x80), B(0xE2), B(0x96), B(0x81), 260, 2, 262};
  StreamDecoder sd(t);
  std::string streamed;
  std::vector<std::string> deltas;
  for (auto id : ids) {
    deltas.push_back(sd.push(id));
    streamed += deltas.back();
  }
  streamed += sd.flush();
  CHECK_EQ(streamed, t.decode(ids, true));
  for (std::size_t i = 1; i <= 7; ++i) CHECK_EQ(deltas[i], std::string(""));  // byte run held back...
  CHECK_EQ(deltas[8], std::string("\xF0\x9F\x98\x80" "\xE2\x96\x81 a"));  // ...until a normal token ends it
  // An invalid byte run must not leak characters that decode() would later turn into U+FFFD.
  const Ids bad = {260, B(0x28), B(0xC3), 262};
  StreamDecoder sb(t);
  std::string out;
  for (auto id : bad) out += sb.push(id);
  out += sb.flush();
  CHECK_EQ(out, t.decode(bad, true));
  CHECK_EQ(out, std::string("a") + FFFD + FFFD + " ab");
}

TEST(tokenizer_chat_template) {
  const std::vector<ChatMessage> m = {{"system", "Be brief."}, {"user", "Hi"}, {"assistant", "Hello!"}, {"user", "  x  "}};
  CHECK_EQ(apply_chat_template(m, true),
           std::string("<|system|>\nBe brief.</s>\n<|user|>\nHi</s>\n<|assistant|>\nHello!</s>\n<|user|>\n  x  </s>\n"
                       "<|assistant|>\n"));
  CHECK_EQ(apply_chat_template({{"user", "Hi"}}, false), std::string("<|user|>\nHi</s>\n"));
  CHECK_EQ(apply_chat_template({}, true), std::string(""));
  CHECK_THROWS(apply_chat_template({{"tool", "x"}}, true), "unknown role");
  const std::string zephyr =
      "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + "
      "eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token "
      "}}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% "
      "endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}";
  CHECK(is_zephyr_template(zephyr));
  std::string other = zephyr;
  other.replace(other.find("<|user|>"), 8, "[INST]");
  CHECK(!is_zephyr_template(other));
}

TEST(sampler_greedy_topk_topp_penalty) {
  const std::vector<float> l = {1.0f, 3.0f, 2.0f, -1.0f};
  Sampler greedy(SamplerConfig{});
  CHECK_EQ(greedy.sample(l.data(), l.size(), {}), 1);
  SamplerConfig k1;
  k1.temperature = 5.0f;
  k1.top_k = 1;
  Sampler s1(k1);
  for (int i = 0; i < 50; ++i) CHECK_EQ(s1.sample(l.data(), l.size(), {}), 1);
  SamplerConfig p0;
  p0.temperature = 1.0f;
  p0.top_p = 1e-6f;  // nucleus keeps only the top token
  Sampler sp(p0);
  for (int i = 0; i < 50; ++i) CHECK_EQ(sp.sample(l.data(), l.size(), {}), 1);
  SamplerConfig rp;
  rp.repetition_penalty = 2.0f;  // 3.0 -> 1.5 < 2.0: token 2 now wins; negative logits are multiplied
  Sampler sr(rp);
  CHECK_EQ(sr.sample(l.data(), l.size(), {1, 1}), 2);
  const std::vector<float> neg = {-1.0f, -1.5f};
  CHECK_EQ(sr.sample(neg.data(), 2, {0}), 1);  // -1 * 2 = -2 < -1.5
}

TEST(sampler_seeded_distribution) {
  const std::vector<float> l = {std::log(0.7f), std::log(0.2f), std::log(0.1f)};
  SamplerConfig c;
  c.temperature = 1.0f;
  c.seed = 42;
  Sampler a(c), b(c);
  std::map<int, int> count;
  bool same = true;
  for (int i = 0; i < 20000; ++i) {
    const auto x = a.sample(l.data(), 3, {});
    same = same && x == b.sample(l.data(), 3, {});
    ++count[x];
  }
  CHECK(same);  // same seed -> same sequence
  CHECK(std::fabs(count[0] / 20000.0 - 0.7) < 0.02);
  CHECK(std::fabs(count[1] / 20000.0 - 0.2) < 0.02);
  CHECK(std::fabs(count[2] / 20000.0 - 0.1) < 0.02);
}

// Stage 8: BPE merges use a heap (O(n log n)). Checked against the plain O(n^2) rule - repeatedly
// merge the lowest-rank adjacent pair, leftmost on ties - on random texts, and timed on a long one
// (Llama has no pre-tokenizer, so a whole text is one "word"; wikitext-2 is 1.3 MB).
TEST(tokenizer_bpe_heap_matches_naive_rule_and_scales) {
  const Tokenizer t = mini();
  // The mini tokenizer's pieces and merges (test_util.h), as ids: S=264 a=265 b=266 c=267.
  const std::map<std::pair<std::int32_t, std::int32_t>, std::pair<int, std::int32_t>> merges = {
      {{266, 267}, {0, 263}}, {{264, 265}, {1, 260}}, {{265, 266}, {2, 261}}, {{260, 266}, {3, 262}},
      {{264, 264}, {4, 259}}, {{259, 264}, {5, 268}}, {{265, 265}, {6, 271}}};
  auto naive = [&](const std::string& text) {
    Ids sym = {264};  // Prepend: the text does not start with a space in these cases
    for (char ch : text) sym.push_back(ch == ' ' ? 264 : 265 + (ch - 'a'));
    for (;;) {
      int best = 1 << 30;
      std::size_t pos = 0;
      for (std::size_t i = 0; i + 1 < sym.size(); ++i) {
        const auto it = merges.find({sym[i], sym[i + 1]});
        if (it != merges.end() && it->second.first < best) {
          best = it->second.first;
          pos = i;
        }
      }
      if (best == 1 << 30) break;
      sym[pos] = merges.at({sym[pos], sym[pos + 1]}).second;
      sym.erase(sym.begin() + static_cast<std::ptrdiff_t>(pos) + 1);
    }
    return sym;
  };
  std::mt19937 rng(5);
  const char alphabet[] = {'a', 'a', 'b', 'c', ' ', ' '};
  int mismatches = 0;
  for (int k = 0; k < 500; ++k) {
    std::string s(1 + rng() % 300, 'a');
    for (char& ch : s) ch = alphabet[rng() % sizeof alphabet];
    s[0] = 'a';
    mismatches += t.encode(s) != naive(s);
  }
  CHECK_EQ(mismatches, 0);
  std::string big(300000, 'a');
  for (char& ch : big) ch = alphabet[rng() % sizeof alphabet];
  big[0] = 'a';
  const auto t0 = std::chrono::steady_clock::now();
  const Ids ids = t.encode(big);
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  CHECK(!ids.empty() && s < 10.0);  // the O(n^2) loop needs minutes here
  std::printf("    300k characters -> %zu tokens in %.3f s\n", ids.size(), s);
}
