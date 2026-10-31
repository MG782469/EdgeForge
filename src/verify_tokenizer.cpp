// EdgeForge - verify-tokenizer: C++ tokenizer vs Hugging Face golden data.
#include "ef/verify_tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "ef/error.h"
#include "ef/format.h"
#include "ef/json.h"
#include "ef/model.h"
#include "ef/tokenizer.h"

namespace fs = std::filesystem;

namespace ef {

namespace {

json::Value load_json(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw Error("cannot open '" + p.string() + "'");
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return json::parse(text);
}

std::vector<std::int32_t> ids_of(const json::Value& arr) {
  std::vector<std::int32_t> v;
  for (const auto& x : arr.items()) v.push_back(static_cast<std::int32_t>(x.as_int64()));
  return v;
}

std::string show_ids(const std::vector<std::int32_t>& v) {
  std::string s = "[";
  for (std::size_t i = 0; i < v.size() && i < 16; ++i) s += (i ? " " : "") + std::to_string(v[i]);
  return s + (v.size() > 16 ? " ...]" : "]");
}

std::string show_text(const std::string& s) {
  return "\"" + json::escape(s.size() > 60 ? s.substr(0, 60) + "..." : s) + "\"";
}

struct Tally {
  std::string name;
  std::size_t ok = 0, total = 0;
};

struct Report {
  int pass = 0, fail = 0;
  std::vector<std::string> details;
  void add(const std::string& detail) {
    if (details.size() < 12) details.push_back(detail);
  }
  void line(const Tally& t) {
    const bool good = t.ok == t.total && t.total > 0;
    std::printf("  %-52s %5zu/%-5zu %s\n", t.name.c_str(), t.ok, t.total, good ? "PASS" : "FAIL");
    if (good) {
      ++pass;
    } else {
      ++fail;
    }
  }
};

}  // namespace

int run_verify_tokenizer(const VerifyTokenizerOptions& opt) {
  const Tokenizer tok = Tokenizer::load(opt.tokenizer_dir);
  const json::Value cases = load_json(opt.golden_dir / "tokenizer_cases.json");
  std::printf("EdgeForge verify-tokenizer: C++ tokenizer vs Hugging Face (%s, legacy=%s)\n",
              cases.at("tokenizer_class").as_string().c_str(),
              cases.find("legacy") && cases.at("legacy").is_bool() ? (cases.at("legacy").as_bool() ? "true" : "false")
                                                                     : "?");
  const char* scheme = tok.prepend_scheme() == PrependScheme::First    ? "first"
                       : tok.prepend_scheme() == PrependScheme::Always ? "always"
                                                                       : "never";
  std::printf("tokenizer: %zu pieces, %zu merges, bos %d, eos %d, prepend scheme '%s'\n\n", tok.vocab_size(),
              tok.merge_count(), tok.bos_id(), tok.eos_id(), scheme);
  Report rep;

  // ---------------- 122 strings
  Tally enc{"encode (no special tokens added)"}, enc_bos{"encode with BOS (HF default)"}, pieces{"token pieces"},
      dec{"decode"}, dec_skip{"decode, skip_special_tokens"};
  for (const auto& c : cases.at("cases").items()) {
    const std::string text = c.at("text").as_string();
    const auto want = ids_of(c.at("ids"));
    const auto got = tok.encode(text, false);
    ++enc.total;
    if (got == want) {
      ++enc.ok;
    } else {
      rep.add("encode " + show_text(text) + ": ours " + show_ids(got) + " vs HF " + show_ids(want));
    }
    ++enc_bos.total;
    if (tok.encode(text, true) == ids_of(c.at("ids_default"))) ++enc_bos.ok;
    ++pieces.total;
    bool same_pieces = c.at("pieces").size() == want.size();
    for (std::size_t i = 0; same_pieces && i < want.size(); ++i) {
      same_pieces = tok.piece(want[i]) == c.at("pieces")[i].as_string();
    }
    if (same_pieces) ++pieces.ok;
    const std::string d = tok.decode(want, false), ds = tok.decode(want, true);
    ++dec.total;
    if (d == c.at("decoded").as_string()) {
      ++dec.ok;
    } else {
      rep.add("decode " + show_ids(want) + ": ours " + show_text(d) + " vs HF " + show_text(c.at("decoded").as_string()));
    }
    ++dec_skip.total;
    if (ds == c.at("decoded_skip_special").as_string()) ++dec_skip.ok;
  }
  for (const auto* t : {&enc, &enc_bos, &pieces, &dec, &dec_skip}) rep.line(*t);

  // ---------------- chat template
  const json::Value chat = load_json(opt.golden_dir / "chat_template_cases.json");
  Tally ct_text{"chat template text"}, ct_ids{"chat template -> ids (BOS + text)"};
  for (const auto& c : chat.at("cases").items()) {
    std::vector<ChatMessage> msgs;
    for (const auto& m : c.at("messages").items()) msgs.push_back({m.at("role").as_string(), m.at("content").as_string()});
    const std::string text = apply_chat_template(msgs, c.at("add_generation_prompt").as_bool(), tok.eos_token());
    ++ct_text.total;
    if (text == c.at("text").as_string()) {
      ++ct_text.ok;
    } else {
      rep.add("template: ours " + show_text(text) + " vs HF " + show_text(c.at("text").as_string()));
    }
    ++ct_ids.total;
    if (tok.encode(c.at("text").as_string(), true) == ids_of(c.at("ids"))) ++ct_ids.ok;
  }
  rep.line(ct_text);
  rep.line(ct_ids);
  {
    const bool zephyr = is_zephyr_template(chat.at("chat_template").as_string()) &&
                        (tok.chat_template().empty() || is_zephyr_template(tok.chat_template()));
    Tally t{"Jinja template is the TinyLlama/Zephyr one we implement", zephyr ? 1u : 0u, 1};
    rep.line(t);
  }

  // ---------------- golden prompts: text <-> ids
  std::vector<fs::path> prompt_dirs;
  if (fs::is_directory(opt.golden_dir / "prompts")) {
    for (const auto& e : fs::directory_iterator(opt.golden_dir / "prompts")) {
      if (fs::exists(e.path() / "manifest.json")) prompt_dirs.push_back(e.path());
    }
  }
  std::sort(prompt_dirs.begin(), prompt_dirs.end());
  Tally p_fmt{"prompts: chat-formatted text"}, p_ids{"prompts: text -> input ids"},
      p_dec{"prompts: HF greedy ids -> text"}, p_dec_sp{"prompts: HF greedy ids -> text (with specials)"};
  struct E2E {
    std::string name;
    std::vector<std::int32_t> ids;
    std::string want;
    std::size_t max_new;
  };
  std::vector<E2E> e2e;
  for (const auto& d : prompt_dirs) {
    const json::Value m = load_json(d / "manifest.json");
    if (!m.find("prompt") || !m.find("formatted")) continue;  // synthetic golden has no text
    const std::string formatted = apply_chat_template({{"user", m.at("prompt").as_string()}}, true, tok.eos_token());
    ++p_fmt.total;
    if (formatted == m.at("formatted").as_string()) ++p_fmt.ok;
    const auto ids = tok.encode(formatted, true);
    ++p_ids.total;
    if (ids == ids_of(m.at("input_ids"))) {
      ++p_ids.ok;
    } else {
      rep.add(d.filename().string() + " ids: ours " + show_ids(ids) + " vs HF " + show_ids(ids_of(m.at("input_ids"))));
    }
    const auto& g = m.at("greedy");
    const auto gids = ids_of(g.at("ids"));
    ++p_dec.total;
    if (tok.decode(gids, true) == g.at("text").as_string()) ++p_dec.ok;
    ++p_dec_sp.total;
    if (tok.decode(gids, false) == g.at("text_with_special").as_string()) ++p_dec_sp.ok;
    e2e.push_back({d.filename().string(), ids, g.at("text").as_string(), static_cast<std::size_t>(g.at("max_new_tokens").as_int64())});
  }
  if (p_fmt.total > 0) {
    for (const auto* t : {&p_fmt, &p_ids, &p_dec, &p_dec_sp}) rep.line(*t);
  }

  // ---------------- end to end: text -> our tokenizer -> our engine -> our decoder
  if (!opt.model_dir.empty() && !e2e.empty()) {
    const Model model = Model::load(opt.model_dir);
    Tally t{"END-TO-END: prompt text -> EdgeForge -> reply == HF reply"};
    std::printf("\n");
    for (const auto& e : e2e) {
      Transformer tr(model, e.ids.size() + e.max_new + 1);
      const GreedyResult r = generate_greedy(tr, e.ids, e.max_new, tok.eos_id());
      const std::string got = tok.decode(r.tokens, true);
      ++t.total;
      const bool same = got == e.want;
      if (same) ++t.ok;
      std::printf("  %-14s %s  %s\n", e.name.c_str(), same ? "same " : "DIFF ", show_text(got).c_str());
      if (!same) rep.add(e.name + ": HF says " + show_text(e.want));
    }
    rep.line(t);
  } else if (opt.model_dir.empty()) {
    std::printf("  %-52s SKIP (pass --model-dir)\n", "END-TO-END text generation vs HF");
  }

  if (!rep.details.empty()) {
    std::printf("\nfirst mismatches:\n");
    for (const auto& d : rep.details) std::printf("  %s\n", d.c_str());
  }
  std::printf("\nRESULT: %s  (%d passed, %d failed)\n", rep.fail ? "FAIL" : "PASS", rep.pass, rep.fail);
  return rep.fail ? 1 : 0;
}

}  // namespace ef
