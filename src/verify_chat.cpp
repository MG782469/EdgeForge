// EdgeForge - Stage 8: verify-chat (see include/ef/verify_chat.h).
#include "ef/verify_chat.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "ef/chat.h"
#include "ef/error.h"
#include "ef/format.h"
#include "ef/json.h"
#include "ef/model.h"
#include "ef/threadpool.h"
#include "ef/tokenizer.h"

namespace fs = std::filesystem;

namespace ef {

namespace {

// A first difference from the reference is accepted for the BF16 engine only where the reference's
// top-1 and top-2 logits are closer than this (our logits match PyTorch to ~1e-4 absolute; fixed
// before the first real-model run).
constexpr double kNearTie = 0.01;

json::Value load_json(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) {
    throw Error("cannot open '" + p.string() +
                "': generate it with tools/hf_chat_reference.py (README, Stage 8 chat)");
  }
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return json::parse(text);
}

std::vector<std::int32_t> ids_of(const json::Value& arr) {
  std::vector<std::int32_t> v;
  for (const auto& x : arr.items()) v.push_back(static_cast<std::int32_t>(x.as_int64()));
  return v;
}

std::vector<double> doubles_of(const json::Value* arr) {
  std::vector<double> v;
  if (arr != nullptr) {
    for (const auto& x : arr->items()) v.push_back(x.as_double());
  }
  return v;
}

std::string one_line(const std::string& s, std::size_t max = 90) {
  std::string r;
  for (char c : s) {
    if (c == '\n') {
      r += "\\n";
    } else if (c == '\r' || c == '\t') {
      r += ' ';
    } else {
      r += c;
    }
  }
  if (r.size() > max) {
    std::size_t cut = max;
    while (cut > 0 && (static_cast<unsigned char>(r[cut]) & 0xC0) == 0x80) --cut;  // UTF-8 boundary
    r = r.substr(0, cut) + "...";
  }
  return "\"" + r + "\"";
}

bool has_marker(const std::string& s) {
  for (const auto& m : StopConfig{}.stop_strings) {
    if (s.find(m) != std::string::npos) return true;
  }
  return false;
}

struct Turn {
  std::string user, text;
  std::vector<std::int32_t> prompt_ids, generated;  // generated includes EOS if the reference stopped there
  std::vector<double> margins, eos_p;
  bool eos = false;
};

struct Run {
  std::string name, kind, system;
  std::vector<Turn> turns;
};

struct Checks {
  int pass = 0, fail = 0;
  void report(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    (ok ? pass : fail) += 1;
  }
};

// Where a raw reply ended, in words.
std::string ending(bool eos, std::size_t n, std::size_t max_new) {
  if (eos) return strf("EOS after %zu tok", n);
  return n >= max_new ? strf("no EOS in %zu tok", n) : strf("%zu tok", n);
}

}  // namespace

int run_verify_chat(const VerifyChatOptions& o) {
  const json::Value g = load_json(o.golden_dir / "chat_generation.json");
  const auto max_new = static_cast<std::size_t>(g.at("max_new_tokens").as_int64());
  std::map<std::string, std::string> systems;
  for (const auto& k : g.at("systems").keys()) systems[k] = g.at("systems").at(k).as_string();
  std::vector<Run> runs;
  for (const auto& r : g.at("runs").items()) {
    Run run{r.at("case").as_string(), r.at("kind").as_string(), r.at("system").as_string(), {}};
    if (!systems.count(run.system)) throw Error("chat golden: unknown system '" + run.system + "'");
    for (const auto& t : r.at("turns").items()) {
      Turn turn;
      turn.user = t.at("user").as_string();
      turn.text = t.at("text").as_string();
      turn.prompt_ids = ids_of(t.at("prompt_ids"));
      turn.generated = ids_of(t.at("generated_ids"));
      turn.eos = t.at("stopped_on_eos").as_bool();
      turn.margins = doubles_of(t.find("margins"));
      turn.eos_p = doubles_of(t.find("eos_p"));
      run.turns.push_back(std::move(turn));
    }
    runs.push_back(std::move(run));
  }

  const bool quant = !o.qmodel.empty();
  if (!quant && o.model_dir.empty()) throw Error("verify-chat needs --model-dir DIR (BF16) or --qmodel FILE.efq");
  const Model model = quant ? Model::load_efq(o.qmodel) : Model::load(o.model_dir);
  const fs::path tok_dir = !o.tokenizer_dir.empty() ? o.tokenizer_dir : o.model_dir;
  if (tok_dir.empty()) throw Error("verify-chat --qmodel needs --tokenizer-dir DIR");
  const Tokenizer tok = Tokenizer::load(tok_dir);
  const std::int32_t eos = tok.eos_id();
  const char* engine = quant ? "4-bit engine" : "BF16 engine";

  std::printf("EdgeForge verify-chat: %s (%s) vs Hugging Face greedy generation\n", engine,
              model.description().c_str());
  const json::Value* ver = g.find("versions");
  std::size_t n_replies = 0;
  for (const Run& r : runs) n_replies += r.turns.size();
  std::printf("  reference: %s, %s, %zu conversations, %zu replies, max %zu new tokens%s\n",
              g.at("reference_dtype").as_string().c_str(),
              ver != nullptr && ver->find("transformers") != nullptr
                  ? ("transformers " + ver->at("transformers").as_string()).c_str()
                  : "?",
              runs.size(), n_replies, max_new, quant ? " (the reference is the BF16 checkpoint)" : "");
  Checks chk;
  chk.report(systems.count("default") && systems["default"] == kDefaultSystemPrompt,
             "the reference used the engine's default system prompt (else re-run tools/hf_chat_reference.py)");

  // ---- 1. raw replay vs Hugging Face
  std::printf("\n1. raw replay (EOS and max-new only), reference history for multi-turn cases:\n");
  ChatOptions raw;
  raw.max_new = max_new;
  raw.stop = StopConfig::none();
  raw.pool = o.pool;
  ChatSession session(model, tok, raw);
  std::size_t n_turns = 0, prompts_same = 0, identical = 0, near_ties = 0, ref_eos = 0, ours_eos = 0;
  // Our raw replies to the default-system runs, by (case, turn): compared with part 3.
  std::map<std::pair<std::string, std::size_t>, TurnStats> raw_default;
  for (const Run& run : runs) {
    std::vector<ChatMessage> hist;
    if (!systems[run.system].empty()) hist.push_back({"system", systems[run.system]});
    for (std::size_t ti = 0; ti < run.turns.size(); ++ti) {
      const Turn& t = run.turns[ti];
      session.set_history(hist);
      TurnStats st;
      session.reply(t.user, {}, &st);
      ++n_turns;
      const bool same_prompt = st.prompt_ids == t.prompt_ids;
      prompts_same += same_prompt;
      std::vector<std::int32_t> ours = st.reply_ids;
      if (st.reason == StopReason::Eos) ours.push_back(eos);
      std::size_t d = 0;
      while (d < ours.size() && d < t.generated.size() && ours[d] == t.generated[d]) ++d;
      const bool same = d == ours.size() && d == t.generated.size();
      identical += same;
      ref_eos += t.eos;
      ours_eos += st.reason == StopReason::Eos;
      std::string verdict;
      if (!same_prompt) {
        verdict = "PROMPT IDS DIFFER";
      } else if (same) {
        verdict = strf("identical %zu/%zu", d, t.generated.size());
      } else {
        const double m = d < t.margins.size() ? t.margins[d] : -1.0;
        const bool tie = m >= 0.0 && m < kNearTie;
        near_ties += tie;
        verdict = strf("same for %zu tokens, then differs (reference top-2 margin %.4f%s)", d, m,
                       tie ? ": near-tie" : quant ? "" : ": NOT a near-tie");
      }
      std::printf("  %-18s %-7s turn %zu: reference %-18s | ours %s\n", run.name.c_str(), run.system.c_str(), ti + 1,
                  ending(t.eos, t.generated.size() - (t.eos ? 1 : 0), max_new).c_str(), verdict.c_str());
      std::fflush(stdout);
      if (run.system == "default") raw_default[{run.name, ti}] = st;
      hist.push_back({"user", t.user});
      hist.push_back({"assistant", t.text});
    }
  }
  chk.report(prompts_same == n_turns,
             strf("prompt ids (chat template + tokenizer) identical to Hugging Face: %zu/%zu turns", prompts_same,
                  n_turns));
  if (quant) {
    std::printf("  4-bit engine: %zu/%zu replies token-identical to the BF16 reference, %zu end with EOS "
                "(reference: %zu) - reported, not a target\n",
                identical, n_turns, ours_eos, ref_eos);
  } else {
    chk.report(identical + near_ties == n_turns,
               strf("greedy reply ids identical to Hugging Face: %zu/%zu turns (+%zu first differing at a near-tie, "
                    "top-2 margin < %.2f)",
                    identical, n_turns, near_ties, kNearTie));
  }

  // ---- 2. how the raw model ends its replies (Hugging Face ids: the model, not our runtime)
  std::printf("\n2. the model's own replies (reference ids, raw greedy, max %zu tokens):\n", max_new);
  std::printf("  %-8s %5s %12s %13s %11s %14s\n", "system", "turns", "end with EOS", "no EOS (max)", "role marker",
              "repeats (x3)");
  std::map<std::string, std::vector<const Run*>> by_system;
  for (const Run& r : runs) by_system[r.system].push_back(&r);
  for (const auto& [sys, list] : by_system) {
    std::size_t n = 0, e = 0, lim = 0, marker = 0, rep = 0;
    for (const Run* r : list) {
      for (const Turn& t : r->turns) {
        ++n;
        e += t.eos;
        lim += !t.eos;
        marker += has_marker(t.text);
        StopConfig rep_only;  // repetition rule alone (a marker must not hide a loop)
        rep_only.stop_strings.clear();
        ReplyGuard rg(tok, rep_only);
        for (std::int32_t id : t.generated) {
          if (id == eos) break;
          rg.push(id);
        }
        rep += rg.reason() == StopReason::Repetition;
      }
    }
    std::printf("  %-8s %5zu %12zu %13zu %11zu %14zu\n", sys.c_str(), n, e, lim, marker, rep);
  }
  std::printf("  replies without EOS, with the step where the model gave EOS its highest probability:\n");
  for (const Run& r : runs) {
    for (std::size_t ti = 0; ti < r.turns.size(); ++ti) {
      const Turn& t = r.turns[ti];
      if (t.eos || t.eos_p.empty()) continue;
      const auto it = std::max_element(t.eos_p.begin(), t.eos_p.end());
      const auto step = static_cast<std::size_t>(it - t.eos_p.begin());
      const std::vector<std::int32_t> head(t.generated.begin(),
                                           t.generated.begin() + static_cast<std::ptrdiff_t>(std::min(step, t.generated.size())));
      std::printf("    %-18s %-7s turn %zu: P(EOS) %.3f after %3zu tok: %s\n", r.name.c_str(), r.system.c_str(), ti + 1,
                  *it, step, one_line(tok.decode(head, true), 70).c_str());
    }
  }

  // ---- 3. default chat settings
  std::printf("\n3. default chat settings: system prompt, role-marker stop, repetition stop (%zu x %zu tokens), "
              "max-new %zu:\n",
              StopConfig{}.repeat_count, StopConfig{}.repeat_span, ChatOptions{}.max_new);
  ChatOptions def;
  def.system_prompt = kDefaultSystemPrompt;
  def.pool = o.pool;
  ChatSession chat(model, tok, def);
  std::map<StopReason, std::size_t> reasons;
  std::size_t n3 = 0, markers_leaked = 0, cut_natural = 0, compared = 0, mismatched = 0, greet = 0, greet_clean = 0,
              marker_in_natural = 0;
  for (const Run& run : runs) {
    if (run.system != "default") continue;
    chat.reset();
    for (std::size_t ti = 0; ti < run.turns.size(); ++ti) {
      TurnStats st;
      const std::string reply = chat.reply(run.turns[ti].user, {}, &st);
      ++n3;
      ++reasons[st.reason];
      markers_leaked += has_marker(reply);
      const bool clean = st.reason == StopReason::Eos || st.reason == StopReason::StopString;
      if (run.kind == "greeting") {
        ++greet;
        greet_clean += clean;
      }
      // Same prompt as the raw replay (always turn 1; later turns if no earlier reply was cut):
      // a reply the model ends with EOS on its own must come out unchanged.
      const auto it = raw_default.find({run.name, ti});
      if (it != raw_default.end() && it->second.prompt_ids == st.prompt_ids &&
          it->second.reason == StopReason::Eos) {
        ++compared;
        // A role marker inside such a reply is a simulated turn: cutting there is the design.
        // Any other early end (a repetition false positive) would cut a real answer.
        if (st.reason == StopReason::StopString) ++marker_in_natural;
        else if (st.reason != StopReason::Eos) ++cut_natural;
        if (st.reason == StopReason::Eos && st.reply_ids != it->second.reply_ids) ++mismatched;
      }
      std::string end = stop_reason_name(st.reason);
      if (st.reason == StopReason::StopString) end += " " + st.stop_detail;
      std::printf("  %-18s turn %zu: %3zu tok, end: %-22s %s\n", run.name.c_str(), ti + 1, st.generated, end.c_str(),
                  one_line(reply).c_str());
      std::fflush(stdout);
    }
  }
  std::printf("  endings:");
  for (const auto& [r, n] : reasons) std::printf(" %s %zu", stop_reason_name(r), n);
  std::printf(" (of %zu replies)\n", n3);
  std::printf("  greetings that ended as one reply (EOS or role marker): %zu/%zu (reported: a model property)\n",
              greet_clean, greet);
  chk.report(markers_leaked == 0, strf("no reply contains a role marker or </s>: %zu/%zu", n3 - markers_leaked, n3));
  chk.report(cut_natural == 0 && mismatched == 0,
             strf("replies the model ends with EOS on its own are never cut short by the repetition stop: %zu/%zu "
                  "unchanged, %zu cut at a role marker inside them (by design)",
                  compared - cut_natural - mismatched - marker_in_natural, compared, marker_in_natural));

  std::printf("\nRESULT: %s  (%d passed, %d failed)\n", chk.fail ? "FAIL" : "PASS", chk.pass, chk.fail);
  return chk.fail ? 1 : 0;
}

}  // namespace ef
