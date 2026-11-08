// EdgeForge - Stage 8: eos-study (see include/ef/eos_study.h).
#include "ef/eos_study.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <set>
#include <sstream>
#include <tuple>

#include "ef/error.h"
#include "ef/format.h"
#include "ef/json.h"
#include "ef/model.h"
#include "ef/threadpool.h"

namespace fs = std::filesystem;

namespace ef {

namespace {

constexpr double kBorderline = 0.002;  // |P(EOS) - tau| below this: BF16 engine vs reference may decide differently

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

std::vector<double> doubles_of(const json::Value& arr) {
  std::vector<double> v;
  for (const auto& x : arr.items()) v.push_back(x.as_double());
  return v;
}

std::string ltrim(std::string s) {
  std::size_t i = 0;
  while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
  return s.substr(i);
}

std::string rtrim(std::string s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
  return s;
}

std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

bool has_marker(const std::string& s) {
  for (const auto& m : StopConfig{}.stop_strings) {
    if (s.find(m) != std::string::npos) return true;
  }
  return false;
}

std::string fnv_key(const std::string& s) {
  std::uint64_t h = 1469598103934665603ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return strf("%016llx", static_cast<unsigned long long>(h));
}

std::string tau_str(double t) { return t == 0.0 ? std::string("off") : strf("%.2f", t); }

std::string q(const std::string& s) { return "\"" + json::escape(s) + "\""; }

std::string preview(const std::string& s, std::size_t max) {
  std::string r;
  for (char c : s) r += c == '\n' ? std::string(" / ") : std::string(1, c);
  if (r.size() > max) {
    std::size_t cut = max;
    while (cut > 0 && (static_cast<unsigned char>(r[cut]) & 0xC0) == 0x80) --cut;
    r = r.substr(0, cut) + "...";
  }
  return r;
}

// ---- code checks ---------------------------------------------------------------------------

std::string norm_lang(const std::string& l) {
  const std::string x = lower(l);
  if (x == "c") return "c";
  if (x == "cpp" || x == "c++" || x == "cc" || x == "cxx") return "cpp";
  if (x == "python" || x == "py" || x == "python3") return "python";
  return x;
}

struct CmdResult {
  int status = -1;
  std::string out;
};

CmdResult run_cmd(const std::string& cmd) {
  CmdResult r;
#if defined(_WIN32)
  FILE* p = _popen(cmd.c_str(), "r");
#else
  FILE* p = popen(cmd.c_str(), "r");
#endif
  if (p == nullptr) return r;
  char buf[512];
  while (std::fgets(buf, sizeof buf, p) != nullptr) r.out += buf;
#if defined(_WIN32)
  r.status = _pclose(p);
#else
  r.status = pclose(p);
#endif
  return r;
}

bool tool_available(const std::string& name) {
#if defined(_WIN32)
  (void)name;
  return false;
#else
  return std::system(("command -v " + name + " > /dev/null 2>&1").c_str()) == 0;
#endif
}

std::string first_error(const std::string& out) {
  std::istringstream in(out);
  std::string line, first;
  while (std::getline(in, line)) {
    if (first.empty() && !rtrim(line).empty()) first = line;
    const auto e = line.find("error");
    if (e != std::string::npos) return rtrim(line.substr(e));
  }
  return rtrim(first);
}

}  // namespace

// ---- replay ----------------------------------------------------------------------------------

EosCut replay_eos_threshold(const Tokenizer& tok, const EosTrace& t, double tau, const StopConfig& guards,
                            std::size_t max_new) {
  if (t.eos_p.size() != t.ids.size() + (t.eos ? 1u : 0u)) {
    throw Error(strf("eos trace: %zu P(EOS) values for %zu steps", t.eos_p.size(), t.ids.size() + (t.eos ? 1u : 0u)));
  }
  EosCut c;
  ReplyGuard g(tok, guards);
  std::size_t k = 0;
  for (;; ++k) {
    if (k >= max_new) {
      c.reason = StopReason::MaxNew;
      break;
    }
    if (k == t.ids.size()) {  // the trajectory's next token is EOS, or it was cut by its own limit
      c.reason = t.eos ? StopReason::Eos : StopReason::MaxNew;
      break;
    }
    const double p = t.eos_p[k];
    if (tau > 0.0 && std::fabs(p - tau) < kBorderline) c.borderline = true;
    if (tau > 0.0 && p >= tau) {  // same order as ChatSession: EOS sampled first, then the threshold
      c.reason = StopReason::EosMinP;
      break;
    }
    g.push(t.ids[k]);
    if (g.reason() != StopReason::None) {
      c.reason = g.reason();
      ++k;
      break;
    }
  }
  c.kept = k;
  g.finish();
  c.retained = g.reply();
  const std::string full = tok.decode(t.ids, /*skip_special=*/true);
  std::size_t lcp = 0;
  while (lcp < full.size() && lcp < c.retained.size() && full[lcp] == c.retained[lcp]) ++lcp;
  c.removed = full.substr(lcp);
  return c;
}

std::vector<CodeBlock> extract_code_blocks(const std::string& text) {
  std::vector<CodeBlock> out;
  std::size_t pos = 0;
  while (true) {
    const std::size_t open = text.find("```", pos);
    if (open == std::string::npos) break;
    const std::size_t eol = text.find('\n', open + 3);
    if (eol == std::string::npos) break;
    CodeBlock b;
    b.lang = norm_lang(rtrim(ltrim(text.substr(open + 3, eol - open - 3))));
    const std::size_t close = text.find("```", eol + 1);
    if (close == std::string::npos) {
      b.code = text.substr(eol + 1);
      b.closed = false;
      out.push_back(std::move(b));
      break;
    }
    b.code = text.substr(eol + 1, close - eol - 1);
    out.push_back(std::move(b));
    pos = close + 3;
  }
  return out;
}

std::string check_code(const std::string& reply, const std::string& case_lang) {
  const auto blocks = extract_code_blocks(reply);
  if (blocks.empty()) return "no code block";
  static int counter = 0;
  std::string verdict = "pass";
  std::size_t checked = 0;
  bool unclosed = false;
  for (const auto& b : blocks) {
    const std::string lang = b.lang.empty() ? norm_lang(case_lang) : b.lang;
    if (lang != "c" && lang != "cpp" && lang != "python") continue;
    unclosed = unclosed || !b.closed;
    const std::string tool = lang == "c" ? "gcc" : lang == "cpp" ? "g++" : "python3";
    if (!tool_available(tool)) return "not checked: " + tool + " not found";
    const fs::path f = fs::temp_directory_path() /
                       strf("edgeforge_eos_%d_%d.%s", static_cast<int>(std::rand() % 100000), counter++,
                            lang == "c" ? "c" : lang == "cpp" ? "cpp" : "py");
    {
      std::ofstream out(f, std::ios::binary);
      if (lang == "c") {
        out << "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <stdbool.h>\n"
               "#include <limits.h>\n#include <math.h>\n#line 1\n";
      } else if (lang == "cpp") {
        out << "#include <bits/stdc++.h>\n#line 1\n";
      }
      out << b.code << "\n";
    }
    std::string cmd;
    if (lang == "c") cmd = "gcc -fsyntax-only -std=c11 -w -x c '" + f.string() + "' 2>&1";
    if (lang == "cpp") cmd = "g++ -fsyntax-only -std=c++17 -w -x c++ '" + f.string() + "' 2>&1";
    if (lang == "python") {
      cmd = "python3 -c \"import ast,sys; ast.parse(open(sys.argv[1], encoding='utf-8').read())\" '" + f.string() +
            "' 2>&1";
    }
    const CmdResult r = run_cmd(cmd);
    std::error_code ec;
    fs::remove(f, ec);
    ++checked;
    if (r.status != 0 && verdict == "pass") verdict = "fail: " + first_error(r.out);
  }
  if (checked == 0) return "no c/c++/python code block";
  if (unclosed) verdict += " (code block not closed)";
  return verdict;
}

// ---- decision --------------------------------------------------------------------------------

EosDecision decide_eos_threshold(const std::vector<EosOutcome>& outcomes, const std::vector<double>& taus,
                                 const std::map<std::string, std::string>& labels) {
  EosDecision d;
  using Slot = std::tuple<std::string, std::string, std::string, std::size_t>;  // engine, case, system, turn
  std::map<Slot, const EosOutcome*> base;
  std::set<std::string> engines;
  for (const auto& o : outcomes) {
    engines.insert(o.engine);
    if (o.tau == 0.0) base[{o.engine, o.case_name, o.system, o.turn}] = &o;
  }
  std::map<std::string, EosLabelNeed> need;
  auto require = [&](const EosOutcome& o) {
    EosLabelNeed& n = need[o.key];
    if (n.key.empty()) {
      n = {o.key, o.case_name, o.kind, o.system, o.retained, o.removed, o.flags, o.turn, {}, {}};
    }
    if (std::find(n.engines.begin(), n.engines.end(), o.engine) == n.engines.end()) n.engines.push_back(o.engine);
    if (std::find(n.taus.begin(), n.taus.end(), o.tau) == n.taus.end()) n.taus.push_back(o.tau);
  };
  auto changed = [&](const EosOutcome& o) {
    const auto it = base.find({o.engine, o.case_name, o.system, o.turn});
    if (it == base.end()) throw Error("eos-study: no threshold-off result for " + o.case_name);
    return o.tau > 0.0 && o.retained != it->second->retained;
  };
  for (const auto& o : outcomes) {
    if (o.kind == "greeting" && o.system == "default") {
      require(o);
    } else if (changed(o)) {
      require(o);
      require(*base.at({o.engine, o.case_name, o.system, o.turn}));
    }
  }
  auto label = [&](const EosOutcome& o) -> std::string {
    const auto it = labels.find(o.key);
    return it == labels.end() ? std::string() : it->second;
  };
  for (const auto& [k, n] : need) {
    d.needed.push_back(n);
    if (!labels.count(k)) ++d.missing;
  }
  // Review order: case order of the study, then system, turn, threshold.
  std::map<std::string, std::size_t> case_index;
  for (const auto& o : outcomes) case_index.emplace(o.case_name, case_index.size());
  std::sort(d.needed.begin(), d.needed.end(), [&](const EosLabelNeed& a, const EosLabelNeed& b) {
    const double ta = *std::min_element(a.taus.begin(), a.taus.end()), tb = *std::min_element(b.taus.begin(), b.taus.end());
    return std::tie(case_index[a.case_name], a.system, a.turn, ta, a.key) <
           std::tie(case_index[b.case_name], b.system, b.turn, tb, b.key);
  });
  auto bad = [](const std::string& l) { return l == "D" || l == "E"; };

  double best_tau = -1.0;
  bool any_pending = false;
  for (double tau : taus) {
    EosDecision::PerTau pt;
    pt.tau = tau;
    bool fail = false, pending = false;
    for (const auto& e : engines) {
      // C1: greetings with the default system prompt, labelled A or B
      std::set<std::string> cases;
      std::size_t good = 0, unl = 0;
      for (const auto& o : outcomes) {
        if (o.engine != e || o.tau != tau || o.kind != "greeting" || o.system != "default" || o.turn != 1) continue;
        cases.insert(o.case_name);
        const std::string l = label(o);
        if (l.empty()) ++unl;
        good += l == "A" || l == "B";
      }
      const std::size_t n = cases.size();
      if (n < 8) {
        pt.c1[e] = strf("FAIL only %zu greeting cases (need 8)", n);
        fail = fail || tau > 0.0;
      } else if (unl > 0) {
        pt.c1[e] = strf("PENDING %zu/%zu labelled, %zu A/B so far", n - unl, n, good);
        if (static_cast<double>(good + unl) < 0.75 * static_cast<double>(n)) {
          pt.c1[e] = strf("FAIL at most %zu/%zu can be A/B", good + unl, n);
          fail = fail || tau > 0.0;
        } else {
          pending = true;
        }
      } else {
        const bool ok = static_cast<double>(good) >= 0.75 * static_cast<double>(n);
        pt.c1[e] = strf("%s %zu/%zu A/B", ok ? "PASS" : "FAIL", good, n);
        fail = fail || (!ok && tau > 0.0);
      }
      // C2 and C3: replies the threshold changed
      std::size_t c2_bad = 0, c3_bad = 0, unl2 = 0, nchanged = 0;
      for (const auto& o : outcomes) {
        if (o.engine != e || o.tau != tau || !changed(o)) continue;
        ++nchanged;
        const EosOutcome& b = *base.at({o.engine, o.case_name, o.system, o.turn});
        const std::string l = label(o), lb = label(b);
        if (l.empty() || lb.empty()) ++unl2;
        if (o.kind != "greeting" && l == "D") ++c2_bad;
        if ((bad(l) && !lb.empty() && !bad(lb)) || (b.code.rfind("pass", 0) == 0 && o.code.rfind("fail", 0) == 0)) {
          ++c3_bad;
        }
      }
      if (tau == 0.0) {
        pt.c2[e] = pt.c3[e] = "-";
      } else {
        pt.c2[e] = c2_bad ? strf("FAIL %zu answer(s) lost content", c2_bad)
                   : unl2 ? strf("PENDING %zu changed replies unlabelled", unl2)
                          : strf("PASS %zu changed replies", nchanged);
        pt.c3[e] = c3_bad ? strf("FAIL %zu got worse", c3_bad)
                   : unl2 ? strf("PENDING %zu changed replies unlabelled", unl2)
                          : strf("PASS %zu changed replies", nchanged);
        fail = fail || c2_bad || c3_bad;
        pending = pending || (unl2 > 0 && !c2_bad && !c3_bad);
      }
    }
    if (tau == 0.0) {
      pt.verdict = "baseline (threshold off)";
    } else if (fail) {
      pt.verdict = "fails";
    } else if (engines.size() < 2) {
      pt.verdict = "incomplete: C4 needs both the BF16 and the 4-bit engine";
      pending = true;
    } else if (pending) {
      pt.verdict = "pending labels";
    } else {
      pt.verdict = "qualifies";
      if (tau > best_tau) best_tau = tau;
    }
    any_pending = any_pending || (tau > 0.0 && pending && !fail);
    d.taus.push_back(pt);
  }
  if (best_tau > 0.0) {
    d.recommendation = strf("--eos-min-p %.2f qualifies (the largest qualifying threshold): it may become the default "
                            "after review.",
                            best_tau);
  } else if (any_pending) {
    d.recommendation = strf("pending: %zu replies still need a human label (or both engines are needed); no "
                            "threshold qualifies yet.",
                            d.missing);
  } else {
    d.recommendation = "no threshold qualifies: keep --eos-min-p opt-in; the default stays off (0).";
  }
  return d;
}

// ---- the study -------------------------------------------------------------------------------

namespace {

struct Case {
  std::string name, kind, lang;
  std::vector<std::string> turns;
};

struct Meta {
  std::string golden, created, versions, model_sha;
  std::size_t max_new = 300, traces = 0;
  std::vector<double> taus;
  std::vector<std::string> system_order, rules;
  std::map<std::string, std::string> systems;
  std::map<std::string, std::size_t> sources;  // trace source -> count
};

std::string hint_flags(const EosCut& c, const std::string& kind, const std::string& code) {
  std::vector<std::string> f;
  if (c.retained.empty()) f.push_back("empty reply");
  if (has_marker(c.removed)) f.push_back("role marker after the cut");
  if (c.reason != StopReason::Eos && c.reason != StopReason::MaxNew && c.removed.rfind("\n\n", 0) == 0) {
    f.push_back("cut at a paragraph break");
  }
  const std::string r = rtrim(c.retained);
  if (!r.empty() && std::string(".!?)\"'`:*").find(r.back()) == std::string::npos &&
      static_cast<unsigned char>(r.back()) < 0x80) {
    f.push_back("ends mid-sentence?");
  }
  std::size_t paragraphs = 0, questions = 0;
  bool speaker = false;
  std::size_t pos = 0;
  while (pos <= r.size()) {
    const std::size_t end = std::min(r.find("\n\n", pos), r.size());
    const std::string p = rtrim(ltrim(r.substr(pos, end - pos)));
    if (!p.empty()) {
      ++paragraphs;
      questions += p.back() == '?';
      const auto colon = p.find(':');
      if (colon != std::string::npos && colon > 0 && colon < 14 &&
          std::all_of(p.begin(), p.begin() + static_cast<std::ptrdiff_t>(colon),
                      [](char ch) { return std::isalpha(static_cast<unsigned char>(ch)) != 0; })) {
        speaker = true;  // "User: ...", "Response: ...", "Asha: ..."
      }
    }
    pos = end + 2;
  }
  if (paragraphs >= 2 && (questions >= 2 || speaker) && kind != "code") f.push_back("dialogue-like paragraphs?");
  if (c.borderline) f.push_back("borderline P(EOS)");
  if (code.find("not closed") != std::string::npos) f.push_back("code block not closed");
  std::string s;
  for (const auto& x : f) s += (s.empty() ? "" : "; ") + x;
  return s;
}

void write_results(const fs::path& p, const Meta& m, const std::vector<EosOutcome>& v) {
  std::ofstream out(p, std::ios::binary);
  out << "{\n \"format\": \"edgeforge-eos-study-results-v1\",\n \"golden\": " << q(m.golden)
      << ",\n \"golden_created\": " << q(m.created) << ",\n \"versions\": " << q(m.versions)
      << ",\n \"model_sha256\": " << q(m.model_sha) << ",\n \"max_new_tokens\": " << m.max_new
      << ",\n \"traces\": " << m.traces << ",\n \"thresholds\": [";
  for (std::size_t i = 0; i < m.taus.size(); ++i) out << (i ? ", " : "") << strf("%.2f", m.taus[i]);
  out << "],\n \"systems\": {";
  for (std::size_t i = 0; i < m.system_order.size(); ++i) {
    out << (i ? ", " : "") << q(m.system_order[i]) << ": " << q(m.systems.at(m.system_order[i]));
  }
  out << "},\n \"sources\": {";
  std::size_t i = 0;
  for (const auto& [k, n] : m.sources) out << (i++ ? ", " : "") << q(k) << ": " << n;
  out << "},\n \"decision_rule\": [";
  for (std::size_t r = 0; r < m.rules.size(); ++r) out << (r ? ",\n  " : "\n  ") << q(m.rules[r]);
  out << "],\n \"outcomes\": [";
  for (std::size_t k = 0; k < v.size(); ++k) {
    const EosOutcome& o = v[k];
    out << (k ? ",\n  " : "\n  ") << "{\"engine\": " << q(o.engine) << ", \"case\": " << q(o.case_name)
        << ", \"kind\": " << q(o.kind) << ", \"lang\": " << q(o.lang) << ", \"system\": " << q(o.system)
        << ", \"turn\": " << o.turn << ", \"tau\": " << strf("%.2f", o.tau) << ", \"reason\": " << q(o.reason)
        << ", \"kept\": " << o.kept << ", \"raw_tokens\": " << o.raw_tokens
        << ", \"raw_eos\": " << (o.raw_eos ? "true" : "false") << ", \"borderline\": " << (o.borderline ? "true" : "false")
        << ", \"marker_in_removed\": " << (o.marker_in_removed ? "true" : "false") << ", \"source\": " << q(o.source)
        << ", \"code\": " << q(o.code) << ", \"flags\": " << q(o.flags) << ", \"key\": " << q(o.key)
        << ", \"retained\": " << q(o.retained) << ", \"removed\": " << q(o.removed) << "}";
  }
  out << "\n ]\n}\n";
}

void read_results(const fs::path& p, Meta& m, std::vector<EosOutcome>& v) {
  const json::Value r = load_json(p);
  if (r.at("format").as_string() != "edgeforge-eos-study-results-v1") throw Error("not an eos-study results file");
  m.golden = r.at("golden").as_string();
  m.created = r.at("golden_created").as_string();
  m.versions = r.at("versions").as_string();
  m.model_sha = r.at("model_sha256").as_string();
  m.max_new = static_cast<std::size_t>(r.at("max_new_tokens").as_int64());
  m.traces = static_cast<std::size_t>(r.at("traces").as_int64());
  m.taus = doubles_of(r.at("thresholds"));
  for (const auto& k : r.at("systems").keys()) {
    m.system_order.push_back(k);
    m.systems[k] = r.at("systems").at(k).as_string();
  }
  for (const auto& k : r.at("sources").keys()) m.sources[k] = static_cast<std::size_t>(r.at("sources").at(k).as_int64());
  for (const auto& x : r.at("decision_rule").items()) m.rules.push_back(x.as_string());
  for (const auto& x : r.at("outcomes").items()) {
    EosOutcome o;
    o.engine = x.at("engine").as_string();
    o.case_name = x.at("case").as_string();
    o.kind = x.at("kind").as_string();
    o.lang = x.at("lang").as_string();
    o.system = x.at("system").as_string();
    o.turn = static_cast<std::size_t>(x.at("turn").as_int64());
    o.tau = x.at("tau").as_double();
    o.reason = x.at("reason").as_string();
    o.kept = static_cast<std::size_t>(x.at("kept").as_int64());
    o.raw_tokens = static_cast<std::size_t>(x.at("raw_tokens").as_int64());
    o.raw_eos = x.at("raw_eos").as_bool();
    o.borderline = x.at("borderline").as_bool();
    o.marker_in_removed = x.at("marker_in_removed").as_bool();
    o.source = x.at("source").as_string();
    o.code = x.at("code").as_string();
    o.flags = x.at("flags").as_string();
    o.key = x.at("key").as_string();
    o.retained = x.at("retained").as_string();
    o.removed = x.at("removed").as_string();
    v.push_back(std::move(o));
  }
}

std::map<std::string, std::string> read_labels(const fs::path& p) {
  std::map<std::string, std::string> labels;
  if (p.empty()) return labels;
  const json::Value r = load_json(p);
  for (const auto& it : r.at("items").items()) {
    const std::string l = it.at("label").as_string();
    if (l.empty()) continue;
    if (l != "A" && l != "B" && l != "C" && l != "D" && l != "E") {
      throw Error("label '" + l + "' for " + it.at("key").as_string() + " is not one of A B C D E");
    }
    labels[it.at("key").as_string()] = l;
  }
  return labels;
}

const char* kRubric[][2] = {{"A", "complete and coherent single reply"},
                            {"B", "understandable but unnecessarily verbose"},
                            {"C", "invented conversation turns"},
                            {"D", "prematurely truncated legitimate answer"},
                            {"E", "incorrect or incoherent answer"}};

void write_todo(const fs::path& p, const EosDecision& d, const std::map<std::string, std::string>& labels) {
  std::ofstream out(p, std::ios::binary);
  out << "{\n \"how\": \"Set every label to one letter of the rubric (judge the reply as a whole; code "
         "correctness is checked separately), then: edgeforge eos-study --results <out>_results.json --labels "
         "this-file\",\n \"rubric\": {";
  for (std::size_t i = 0; i < 5; ++i) out << (i ? ", " : "") << q(kRubric[i][0]) << ": " << q(kRubric[i][1]);
  out << "},\n \"items\": [";
  std::size_t n = 0;
  for (const auto& it : d.needed) {
    const auto l = labels.find(it.key);
    std::string taus, engines;
    for (double t : it.taus) taus += (taus.empty() ? "" : ", ") + tau_str(t);
    for (const auto& e : it.engines) engines += (engines.empty() ? "" : ", ") + e;
    out << (n++ ? ",\n  " : "\n  ") << "{\"key\": " << q(it.key) << ", \"case\": " << q(it.case_name)
        << ", \"kind\": " << q(it.kind) << ", \"system\": " << q(it.system) << ", \"turn\": " << it.turn
        << ", \"thresholds\": " << q(taus) << ", \"engines\": " << q(engines) << ", \"hints\": " << q(it.flags)
        << ",\n   \"reply\": " << q(it.retained) << ",\n   \"removed_after_reply\": " << q(preview(it.removed, 400))
        << ",\n   \"label\": " << q(l == labels.end() ? std::string() : l->second) << "}";
  }
  out << "\n ]\n}\n";
}

std::string fence(const std::string& s) { return "~~~~text\n" + s + (s.empty() || s.back() != '\n' ? "\n" : "") + "~~~~\n"; }

void write_report(const fs::path& p, const Meta& m, const std::vector<EosOutcome>& v, const EosDecision& d,
                  const std::map<std::string, std::string>& labels) {
  std::ofstream md(p, std::ios::binary);
  std::set<std::string> engines;
  for (const auto& o : v) engines.insert(o.engine);
  auto ename = [](const std::string& e) { return e == "bf16" ? std::string("BF16") : std::string("4-bit"); };
  md << "# EdgeForge EOS-threshold study\n\n";
  md << "Can `--eos-min-p` (end the turn when the model gives EOS at least this probability) end chat replies "
        "better without damaging real answers? Generated by `edgeforge eos-study`.\n\n";
  md << "```\n";
  md << "reference : " << m.golden << " (Hugging Face greedy traces, created " << m.created << ")\n";
  md << "versions  : " << m.versions << "\nmodel     : sha256 " << m.model_sha << "\n";
  md << "settings  : greedy, role-marker + repetition stops on, max " << m.max_new
     << " new tokens; only the threshold changes\n";
  md << "traces    : " << m.traces << " reference traces checked (P(EOS) present at every step)\nused      :";
  for (const auto& [k, n] : m.sources) md << " " << k << " " << n << ";";
  md << "\nengines   :";
  for (const auto& e : engines) md << " " << ename(e);
  md << "\n```\n\n";
  md << "## Decision rule (fixed before the results)\n\n";
  for (const auto& r : m.rules) md << "- " << r << "\n";
  md << "\nRubric: ";
  for (std::size_t i = 0; i < 5; ++i) md << (i ? ", " : "") << "**" << kRubric[i][0] << "** " << kRubric[i][1];
  md << ".\n\n## Result\n\n**" << d.recommendation << "**\n\n";
  md << "Human labels: " << d.needed.size() - d.missing << " of " << d.needed.size()
     << " needed replies labelled. Automatic hints in the tables are for the reviewer only.\n\n";
  md << "| threshold | C1 greetings A/B |";
  md << " C2 no lost content | C3 nothing worse | verdict |\n|---|---|---|---|---|\n";
  for (const auto& pt : d.taus) {
    auto cell = [&](const std::map<std::string, std::string>& c) {
      std::string s;
      for (const auto& e : engines) s += (s.empty() ? "" : "<br>") + ename(e) + ": " + (c.count(e) ? c.at(e) : "-");
      return s;
    };
    md << "| " << tau_str(pt.tau) << " | " << cell(pt.c1) << " | " << cell(pt.c2) << " | " << cell(pt.c3) << " | "
       << pt.verdict << " |\n";
  }

  // How replies end, per engine and threshold
  md << "\n## How replies end\n\nAll cases, both system prompts. *changed* = reply differs from the threshold-off "
        "reply; *borderline* = a step's P(EOS) within 0.002 of the threshold.\n\n";
  for (const auto& e : engines) {
    md << "**" << ename(e) << "**\n\n| threshold | EOS | EOS probability | role marker | repetition | max-new | changed | "
          "borderline | code compiles |\n|---|---|---|---|---|---|---|---|---|\n";
    for (double tau : m.taus) {
      std::map<std::string, std::size_t> c;
      std::size_t changed = 0, border = 0, code_ok = 0, code_n = 0;
      for (const auto& o : v) {
        if (o.engine != e || o.tau != tau) continue;
        ++c[o.reason];
        border += o.borderline;
        if (o.kind == "code") {
          ++code_n;
          code_ok += o.code.rfind("pass", 0) == 0;
        }
        for (const auto& b : v) {
          if (b.engine == e && b.tau == 0.0 && b.case_name == o.case_name && b.system == o.system && b.turn == o.turn &&
              b.retained != o.retained) {
            ++changed;
          }
        }
      }
      md << "| " << tau_str(tau) << " | " << c["EOS"] << " | " << c["EOS probability"] << " | " << c["role marker"]
         << " | " << c["repetition"] << " | " << c["max-new limit"] << " | " << changed << " | " << border << " | "
         << code_ok << "/" << code_n << " |\n";
    }
    md << "\n";
  }

  // Per case
  md << "## Per prompt\n\nEach row groups the thresholds that give the same reply. *removed* = what the raw "
        "trajectory continues with after the reply.\n";
  std::vector<std::string> case_order;
  for (const auto& o : v) {
    if (std::find(case_order.begin(), case_order.end(), o.case_name) == case_order.end()) case_order.push_back(o.case_name);
  }
  for (const auto& cn : case_order) {
    for (const auto& sys : m.system_order) {
      std::size_t turns = 0;
      std::string kind;
      for (const auto& o : v) {
        if (o.case_name == cn && o.system == sys) {
          turns = std::max(turns, o.turn);
          kind = o.kind;
        }
      }
      for (std::size_t t = 1; t <= turns; ++t) {
        md << "\n### " << cn << " (" << kind << ") · system: " << sys << " · turn " << t << "\n";
        for (const auto& e : engines) {
          std::vector<const EosOutcome*> rows;
          for (const auto& o : v) {
            if (o.engine == e && o.case_name == cn && o.system == sys && o.turn == t) rows.push_back(&o);
          }
          if (rows.empty()) continue;
          const EosOutcome& b = *rows.front();
          md << "\n**" << ename(e) << "** - " << b.source << "; raw reply "
             << (b.raw_eos ? strf("ends with EOS after %zu tokens", b.raw_tokens)
                           : strf("has no EOS within %zu tokens", b.raw_tokens))
             << "\n\n| thresholds | end | tokens | label | code | hints |\n|---|---|---|---|---|---|\n";
          std::vector<std::pair<std::string, std::vector<const EosOutcome*>>> groups;  // by retained text
          for (const auto* o : rows) {
            auto it = std::find_if(groups.begin(), groups.end(), [&](const auto& g) { return g.first == o->key + o->reason; });
            if (it == groups.end()) {
              groups.push_back({o->key + o->reason, {o}});
            } else {
              it->second.push_back(o);
            }
          }
          for (const auto& [k, g] : groups) {
            std::string ts;
            for (const auto* o : g) ts += (ts.empty() ? "" : ", ") + tau_str(o->tau);
            const auto l = labels.find(g.front()->key);
            md << "| " << ts << " | " << g.front()->reason << " | " << g.front()->kept << " | "
               << (l != labels.end() ? l->second : std::string("? `") + g.front()->key.substr(0, 8) + "`") << " | "
               << (kind == "code" ? g.front()->code : std::string()) << " | " << g.front()->flags << " |\n";
          }
          for (const auto& [k, g] : groups) {
            std::string ts;
            for (const auto* o : g) ts += (ts.empty() ? "" : ", ") + tau_str(o->tau);
            md << "\n<details><summary>" << ts << ": reply (" << g.front()->kept << " tokens)"
               << (g.front()->removed.empty() ? "" : " and removed text") << "</summary>\n\n"
               << fence(g.front()->retained);
            if (!rtrim(g.front()->removed).empty()) md << "removed:\n\n" << fence(preview(ltrim(g.front()->removed), 600));
            md << "\n</details>\n";
          }
        }
      }
    }
  }
}

}  // namespace

int run_eos_study(const EosStudyOptions& o) {
  Meta m;
  std::vector<EosOutcome> outcomes;
  const auto labels = read_labels(o.labels);

  if (!o.results.empty()) {  // decide mode: no model runs
    read_results(o.results, m, outcomes);
    std::printf("EdgeForge eos-study: %zu results from %s, %zu labels\n", outcomes.size(), o.results.string().c_str(),
                labels.size());
  } else {
    if (o.golden.empty()) throw Error("eos-study needs --golden FILE (tools/hf_chat_reference.py --study) or --results FILE");
    if (o.model_dir.empty() && o.tokenizer_dir.empty()) throw Error("eos-study needs --model-dir DIR (BF16 + tokenizer)");
    const json::Value g = load_json(o.golden);
    m.golden = o.golden.filename().string();
    m.created = g.at("created_utc").as_string();
    const json::Value& ver = g.at("versions");
    for (const auto& k : ver.keys()) m.versions += (m.versions.empty() ? "" : ", ") + k + " " + ver.at(k).as_string();
    m.model_sha = g.at("model_safetensors_sha256").as_string();
    m.max_new = static_cast<std::size_t>(g.at("max_new_tokens").as_int64());
    m.taus = doubles_of(g.at("thresholds"));
    if (m.taus.empty() || m.taus.front() != 0.0) throw Error("eos-study: the first threshold must be 0 (off)");
    for (const auto& x : g.at("decision_rule").items()) m.rules.push_back(x.as_string());
    for (const auto& k : g.at("systems").keys()) {
      m.system_order.push_back(k);
      m.systems[k] = g.at("systems").at(k).as_string();
    }
    if (!m.systems.count("default") || m.systems["default"] != kDefaultSystemPrompt) {
      throw Error("eos-study: the reference's 'default' system prompt is not the engine's default");
    }
    std::vector<Case> cases;
    for (const auto& c : g.at("cases").items()) {
      Case k{c.at("name").as_string(), c.at("kind").as_string(), c.find("lang") ? c.at("lang").as_string() : "", {}};
      for (const auto& t : c.at("turns").items()) k.turns.push_back(t.as_string());
      cases.push_back(std::move(k));
    }
    const fs::path tok_dir = !o.tokenizer_dir.empty() ? o.tokenizer_dir : o.model_dir;
    const Tokenizer tok = Tokenizer::load(tok_dir);
    const std::int32_t eos = tok.eos_id();

    // Hugging Face traces, by prompt ids. Every trace must carry P(EOS) at every step.
    std::map<std::vector<std::int32_t>, EosTrace> golden;
    for (const auto& r : g.at("runs").items()) {
      for (const auto& t : r.at("turns").items()) {
        EosTrace tr;
        tr.ids = ids_of(t.at("generated_ids"));
        tr.eos = t.at("stopped_on_eos").as_bool();
        if (tr.eos) {
          if (tr.ids.empty() || tr.ids.back() != eos) throw Error("golden trace: stopped_on_eos without EOS");
          tr.ids.pop_back();
        }
        tr.eos_p = doubles_of(t.at("eos_p"));
        if (tr.eos_p.size() != tr.ids.size() + (tr.eos ? 1u : 0u)) {
          throw Error("golden trace " + r.at("case").as_string() + ": P(EOS) missing for some steps");
        }
        if (!tr.eos && tr.ids.size() < m.max_new) {
          throw Error("golden trace " + r.at("case").as_string() + ": shorter than max_new_tokens without EOS");
        }
        tr.source = "Hugging Face reference";
        golden[ids_of(t.at("prompt_ids"))] = std::move(tr);
        ++m.traces;
      }
    }
    std::printf("EdgeForge eos-study: %zu cases x %zu system prompts x %zu thresholds; %zu reference traces checked "
                "(P(EOS) at every step)\n",
                cases.size(), m.system_order.size(), m.taus.size(), m.traces);

    struct Engine {
      std::string id;
      fs::path path;
      bool quant;
      std::unique_ptr<Model> model;
      std::unique_ptr<ChatSession> chat;
      std::map<std::vector<std::int32_t>, EosTrace> cache;
    };
    std::vector<Engine> engines;
    if (!o.model_dir.empty()) engines.push_back({"bf16", o.model_dir, false, nullptr, nullptr, {}});
    if (!o.qmodel.empty()) engines.push_back({"q4", o.qmodel, true, nullptr, nullptr, {}});

    std::map<std::pair<std::string, std::string>, std::string> code_cache;
    for (Engine& e : engines) {
      std::printf("\n%s engine:\n", e.quant ? "4-bit" : "BF16");
      auto trace_for = [&](const std::vector<std::int32_t>& prompt, const std::vector<ChatMessage>& hist,
                           const std::string& user, const std::string& what) -> const EosTrace& {
        if (!e.quant) {
          const auto it = golden.find(prompt);
          if (it != golden.end()) return it->second;
        }
        const auto c = e.cache.find(prompt);
        if (c != e.cache.end()) return c->second;
        if (!e.model) {
          e.model = std::make_unique<Model>(e.quant ? Model::load_efq(e.path) : Model::load(e.path));
          ChatOptions raw;
          raw.max_new = m.max_new;
          raw.stop = StopConfig::none();
          raw.pool = o.pool;
          e.chat = std::make_unique<ChatSession>(*e.model, tok, raw);
        }
        const auto t0 = std::chrono::steady_clock::now();
        e.chat->set_history(hist);
        TurnStats st;
        e.chat->reply(user, {}, &st);
        if (st.prompt_ids != prompt) throw Error("eos-study: engine prompt ids differ from the rendered prompt");
        EosTrace tr;
        tr.ids = st.reply_ids;
        tr.eos = st.reason == StopReason::Eos;
        tr.eos_p = st.eos_p_steps;
        tr.source = e.quant ? "4-bit engine" : "BF16 engine";
        std::printf("  trace %-34s %3zu tok, %s (%.1f s)\n", what.c_str(), tr.ids.size(), tr.eos ? "EOS" : "no EOS",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        std::fflush(stdout);
        return e.cache.emplace(prompt, std::move(tr)).first->second;
      };
      std::map<std::string, std::set<std::vector<std::int32_t>>> sources;  // distinct traces per source
      for (double tau : m.taus) {
        for (const Case& c : cases) {
          for (const auto& sys : m.system_order) {
            std::vector<ChatMessage> hist;
            if (!m.systems[sys].empty()) hist.push_back({"system", m.systems[sys]});
            for (std::size_t ti = 0; ti < c.turns.size(); ++ti) {
              std::vector<ChatMessage> msgs = hist;
              msgs.push_back({"user", c.turns[ti]});
              const auto prompt = tok.encode(apply_chat_template(msgs, true, tok.eos_token()), true, true);
              const EosTrace& tr = trace_for(prompt, hist, c.turns[ti], c.name + "/" + sys + " turn " + std::to_string(ti + 1));
              const EosCut cut = replay_eos_threshold(tok, tr, tau, StopConfig{}, m.max_new);
              EosOutcome out;
              out.engine = e.id;
              out.case_name = c.name;
              out.kind = c.kind;
              out.lang = c.lang;
              out.system = sys;
              out.turn = ti + 1;
              out.tau = tau;
              out.reason = stop_reason_name(cut.reason);
              out.kept = cut.kept;
              out.raw_tokens = tr.ids.size();
              out.raw_eos = tr.eos;
              out.borderline = cut.borderline;
              out.marker_in_removed = has_marker(cut.removed);
              out.retained = cut.retained;
              out.removed = cut.removed;
              out.source = tr.source;
              if (c.kind == "code") {
                const auto key = std::make_pair(cut.retained, c.lang);
                auto it = code_cache.find(key);
                if (it == code_cache.end()) it = code_cache.emplace(key, check_code(cut.retained, c.lang)).first;
                out.code = it->second;
              }
              out.flags = hint_flags(cut, c.kind, out.code);
              out.key = fnv_key(c.name + "|" + sys + "|" + std::to_string(ti + 1) + "|" + cut.retained);
              sources[(e.quant ? "4-bit: " : "BF16: ") + tr.source].insert(prompt);
              outcomes.push_back(std::move(out));
              hist.push_back({"user", c.turns[ti]});
              hist.push_back({"assistant", cut.retained});
            }
          }
        }
        std::size_t changed = 0, n = 0;
        for (const auto& x : outcomes) {
          if (x.engine != e.id || x.tau != tau) continue;
          ++n;
          for (const auto& b : outcomes) {
            changed += b.engine == e.id && b.tau == 0.0 && b.case_name == x.case_name && b.system == x.system &&
                       b.turn == x.turn && b.retained != x.retained;
          }
        }
        std::printf("  threshold %-4s: %zu replies, %zu differ from threshold off\n", tau_str(tau).c_str(), n, changed);
      }
      for (const auto& [k, n] : sources) m.sources[k] = n.size();
    }
  }

  const EosDecision d = decide_eos_threshold(outcomes, m.taus, labels);
  const fs::path out_md = o.out + ".md", out_res = o.out + "_results.json";
  fs::path out_todo = o.out + "_labels_todo.json";
  if (!out_md.parent_path().empty()) fs::create_directories(out_md.parent_path());
  if (o.results.empty()) write_results(out_res, m, outcomes);
  if (fs::exists(out_todo) && fs::absolute(out_todo) != fs::absolute(o.labels)) {
    out_todo = o.out + "_labels_todo.new.json";  // never overwrite a file someone may be labelling
  }
  write_todo(out_todo, d, labels);
  write_report(out_md, m, outcomes, d, labels);

  std::printf("\n  threshold  verdict\n");
  for (const auto& pt : d.taus) std::printf("  %-9s  %s\n", tau_str(pt.tau).c_str(), pt.verdict.c_str());
  std::printf("\n%s\n", d.recommendation.c_str());
  std::printf("labels: %zu of %zu needed replies labelled -> %s\n", d.needed.size() - d.missing, d.needed.size(),
              out_todo.string().c_str());
  std::printf("wrote %s%s\n", out_md.string().c_str(), o.results.empty() ? (" and " + out_res.string()).c_str() : "");
  return 0;
}

}  // namespace ef
