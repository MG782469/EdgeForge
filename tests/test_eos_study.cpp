// EdgeForge - Stage 8 tests: the EOS-threshold study (replay, code checks, predeclared decision).
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "ef/eos_study.h"
#include "ef/tokenizer.h"
#include "test_framework.h"
#include "test_util.h"

using namespace ef;

namespace {

using Ids = std::vector<std::int32_t>;
Tokenizer mini() { return Tokenizer::from_json(ef_test::mini_tokenizer_json(), ef_test::mini_tokenizer_config()); }
Ids text_ids(const Tokenizer& t, const std::string& s) { return t.encode(s, false, false); }

EosTrace trace(const Ids& ids, std::vector<double> eos_p, bool eos) {
  EosTrace t;
  t.ids = ids;
  t.eos_p = std::move(eos_p);
  t.eos = eos;
  return t;
}

bool have(const char* tool) { return std::system((std::string("command -v ") + tool + " > /dev/null 2>&1").c_str()) == 0; }

}  // namespace

TEST(eos_replay_follows_chat_session_order) {
  const Tokenizer tok = mini();
  const Ids ids = text_ids(tok, "abc abc. hi");  // 9 tokens or so; exact count below
  const std::size_t n = ids.size();
  std::vector<double> p(n + 1, 0.01);
  p[4] = 0.12;
  p[6] = 0.31;
  p[n] = 0.9;  // the trajectory ends with EOS
  const EosTrace t = trace(ids, p, true);
  // Threshold off: the natural EOS.
  EosCut c = replay_eos_threshold(tok, t, 0.0, StopConfig{}, 300);
  CHECK(c.reason == StopReason::Eos);
  CHECK_EQ(c.kept, n);
  CHECK_EQ(c.retained, tok.decode(ids, true));
  CHECK_EQ(c.removed, std::string());
  // 0.10 stops before token 4, 0.30 before token 6, 0.95 never (EOS comes first).
  c = replay_eos_threshold(tok, t, 0.10, StopConfig{}, 300);
  CHECK(c.reason == StopReason::EosMinP);
  CHECK_EQ(c.kept, std::size_t{4});
  CHECK_EQ(c.retained, tok.decode(Ids(ids.begin(), ids.begin() + 4), true));
  CHECK_EQ(c.retained + c.removed, tok.decode(ids, true));
  c = replay_eos_threshold(tok, t, 0.30, StopConfig{}, 300);
  CHECK(c.reason == StopReason::EosMinP && c.kept == 6);
  c = replay_eos_threshold(tok, t, 0.95, StopConfig{}, 300);
  CHECK(c.reason == StopReason::Eos && c.kept == n);
  // Borderline: a P(EOS) within 0.002 of the threshold.
  CHECK(replay_eos_threshold(tok, t, 0.121, StopConfig{}, 300).borderline);
  CHECK(!replay_eos_threshold(tok, t, 0.13, StopConfig{}, 300).borderline);
  // max-new before the end of the trajectory; a trajectory without EOS ends at its own limit.
  c = replay_eos_threshold(tok, t, 0.0, StopConfig{}, 3);
  CHECK(c.reason == StopReason::MaxNew && c.kept == 3);
  const EosTrace open = trace(ids, std::vector<double>(n, 0.0), false);
  CHECK(replay_eos_threshold(tok, open, 0.2, StopConfig{}, 300).reason == StopReason::MaxNew);
  // Inconsistent traces are rejected.
  CHECK_THROWS(replay_eos_threshold(tok, trace(ids, std::vector<double>(n, 0.0), true), 0.1, StopConfig{}, 300),
               "P(EOS)");
}

TEST(eos_replay_role_marker_and_repetition_still_apply) {
  const Tokenizer tok = mini();
  const Ids ids = text_ids(tok, "ab\n<|user|>\nhi abc");
  const std::size_t n = ids.size();
  // A marker before any threshold step: the guard cuts there.
  EosCut c = replay_eos_threshold(tok, trace(ids, std::vector<double>(n + 1, 0.0), true), 0.2, StopConfig{}, 300);
  CHECK(c.reason == StopReason::StopString);
  CHECK_EQ(c.retained, std::string("ab"));
  CHECK(c.removed.find("<|user|>") != std::string::npos);
  // The threshold firing first (step 1) wins.
  std::vector<double> p(n + 1, 0.0);
  p[1] = 0.5;
  c = replay_eos_threshold(tok, trace(ids, p, true), 0.2, StopConfig{}, 300);
  CHECK(c.reason == StopReason::EosMinP && c.kept == 1);
  // A loop: the repetition stop, as in chat.
  Ids loop;
  for (int k = 0; k < 20; ++k) loop.insert(loop.end(), {265, 266, 267});
  c = replay_eos_threshold(tok, trace(loop, std::vector<double>(loop.size(), 0.0), false), 0.0, StopConfig{}, 300);
  CHECK(c.reason == StopReason::Repetition && c.kept == 18);
  CHECK_EQ(c.retained, std::string("abc"));
}

TEST(eos_code_blocks_and_syntax_check) {
  const std::string reply =
      "Here you go:\n\n```cpp\nint findMax(int arr[], int n) {\n  int m = arr[0];\n  for (int I = 1; I < n; i++)\n"
      "    if (arr[i] > m) m = arr[i];\n  return m;\n}\n```\n\nAnd in Python:\n```python\ndef f(x):\n    return x\n```\n";
  const auto blocks = extract_code_blocks(reply);
  CHECK_EQ(blocks.size(), std::size_t{2});
  CHECK_EQ(blocks[0].lang, std::string("cpp"));
  CHECK_EQ(blocks[1].lang, std::string("python"));
  CHECK(blocks[0].closed && blocks[1].closed);
  const auto cut = extract_code_blocks("```c\nint x = 1;\n");
  CHECK(cut.size() == 1 && !cut[0].closed && cut[0].code == "int x = 1;\n");
  CHECK_EQ(check_code("no code here", "cpp"), std::string("no code block"));
  if (have("g++") && have("gcc") && have("python3")) {
    const std::string bad = check_code(reply, "cpp");
    CHECK(bad.rfind("fail: error:", 0) == 0);
    CHECK(bad.find("not declared") != std::string::npos);  // the I / i bug seen on the laptop
    CHECK_EQ(check_code("```cpp\nint findMax(int a[], int n) {\n  int m = a[0];\n  for (int i = 1; i < n; i++)\n"
                        "    if (a[i] > m) m = a[i];\n  return m;\n}\n```",
                        "cpp"),
             std::string("pass"));
    CHECK_EQ(check_code("```\nint larger(int a, int b) { return a > b ? a : b; }\n```", "c"), std::string("pass"));
    CHECK(check_code("```python\ndef f(:\n    pass\n```", "python").rfind("fail", 0) == 0);
    CHECK(check_code("```c\nint main(void) { printf(\"%d\", 1); return 0; }\n", "c") == "pass (code block not closed)");
  }
}

TEST(eos_decision_rule_is_the_predeclared_one) {
  // 8 greetings + one factual, one code case; two engines; thresholds off and 0.2.
  std::vector<EosOutcome> v;
  auto add = [&](const std::string& eng, const std::string& c, const std::string& kind, double tau, const std::string& text,
                 const std::string& code = "") {
    EosOutcome o;
    o.engine = eng;
    o.case_name = c;
    o.kind = kind;
    o.system = "default";
    o.tau = tau;
    o.retained = text;
    o.code = code;
    o.key = c + "|" + text;
    v.push_back(o);
  };
  for (const char* e : {"bf16", "q4"}) {
    for (int i = 0; i < 8; ++i) {
      const std::string c = "greet" + std::to_string(i);
      add(e, c, "greeting", 0.0, "Hello! How are you? I'm fine. How are you?");
      add(e, c, "greeting", 0.2, i < 6 ? "Hello!" : "Hello! How are you? I'm fine. How are you?");
    }
    add(e, "fact", "factual", 0.0, "Paris. It is a large city.");
    add(e, "fact", "factual", 0.2, "Paris.");
    add(e, "code", "code", 0.0, "```c\nok\n```", "pass");
    add(e, "code", "code", 0.2, "```c\nok\n```", "pass");
  }
  const std::vector<double> taus = {0.0, 0.2};
  // No labels: everything pending; needed = 8 greeting texts x 2 + fact (both texts).
  EosDecision d = decide_eos_threshold(v, taus, {});
  CHECK_EQ(d.needed.size(), std::size_t{2 * 8 - 2 + 2});  // greet6/7 keep the same text at 0.2
  CHECK_EQ(d.missing, d.needed.size());
  CHECK_EQ(d.taus[1].verdict, std::string("pending labels"));
  CHECK(d.recommendation.rfind("pending", 0) == 0);
  // Labels: 6/8 greetings A at 0.2 (75%), fact cut is fine (A).
  std::map<std::string, std::string> L;
  for (int i = 0; i < 8; ++i) {
    L["greet" + std::to_string(i) + "|Hello! How are you? I'm fine. How are you?"] = "C";
    L["greet" + std::to_string(i) + "|Hello!"] = "A";
  }
  L["fact|Paris. It is a large city."] = "A";
  L["fact|Paris."] = "A";
  d = decide_eos_threshold(v, taus, L);
  CHECK_EQ(d.missing, std::size_t{0});
  CHECK_EQ(d.taus[1].verdict, std::string("qualifies"));
  CHECK(d.taus[1].c1.at("bf16").rfind("PASS 6/8", 0) == 0);
  CHECK_EQ(d.taus[0].c1.at("q4"), std::string("FAIL 0/8 A/B"));  // threshold off: no single replies
  CHECK(d.recommendation.find("0.20 qualifies") != std::string::npos);
  // C2: the factual answer lost content -> fails.
  L["fact|Paris."] = "D";
  d = decide_eos_threshold(v, taus, L);
  CHECK_EQ(d.taus[1].verdict, std::string("fails"));
  CHECK(d.taus[1].c2.at("bf16").rfind("FAIL", 0) == 0);
  CHECK(d.recommendation.rfind("no threshold qualifies", 0) == 0);
  L["fact|Paris."] = "A";
  // C1: only 5/8 -> fails.
  L["greet5|Hello!"] = "E";
  d = decide_eos_threshold(v, taus, L);
  CHECK_EQ(d.taus[1].verdict, std::string("fails"));
  CHECK(d.taus[1].c1.at("q4").rfind("FAIL 5/8", 0) == 0);
  CHECK(d.taus[1].c3.at("q4").rfind("FAIL", 0) == 0);  // E where the threshold-off reply was C: worse
  L["greet5|Hello!"] = "A";
  // C3: code that compiled stops compiling (automatic, no label needed).
  for (auto& o : v) {
    if (o.case_name == "code" && o.tau == 0.2) {
      o.retained = "```c\nok";
      o.key = "code|cut";
      o.code = "fail: error: x";
    }
  }
  L["code|cut"] = "B";
  L["code|```c\nok\n```"] = "A";
  d = decide_eos_threshold(v, taus, L);
  CHECK_EQ(d.taus[1].verdict, std::string("fails"));
  CHECK(d.taus[1].c3.at("bf16").rfind("FAIL 1 got worse", 0) == 0);
  // C4: one engine only -> never qualifies.
  std::vector<EosOutcome> one;
  for (const auto& o : v) {
    if (o.engine == "bf16" && o.case_name != "code") one.push_back(o);
  }
  d = decide_eos_threshold(one, taus, L);
  CHECK(d.taus[1].verdict.rfind("incomplete", 0) == 0);
  // Fewer than 8 greeting cases -> C1 fails whatever the labels.
  std::vector<EosOutcome> few;
  for (const auto& o : one) {
    if (o.case_name != "greet7") few.push_back(o);
  }
  CHECK(decide_eos_threshold(few, taus, L).taus[1].c1.at("bf16").rfind("FAIL only 7", 0) == 0);
}
