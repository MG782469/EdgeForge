// EdgeForge - Stage 8: `edgeforge eos-study`, can an EOS-probability threshold end chat replies
// better without damaging real answers?
//
// Inputs: the predeclared cases (tests/fixtures/chat_eos_study.json) and Hugging Face's greedy
// traces for them (tools/hf_chat_reference.py --study -> tests/golden/chat_eos_study.json: token
// ids and P(EOS) at every step). For each threshold, each reply is replayed with the chat
// defaults (role-marker and repetition stops on) plus `--eos-min-p threshold`:
//   - BF16: offline from the Hugging Face trace (our BF16 engine reproduces Hugging Face's greedy
//     tokens, verify-chat 28/28; within 0.002 of a threshold a step is flagged "borderline").
//     Only a multi-turn history that differs from the reference's (an earlier reply was cut) is
//     run on the BF16 engine.
//   - 4-bit: the 4-bit engine's own traces (its P(EOS) differs from BF16's), then the same replay.
// Code replies are compiled (gcc / g++ -fsyntax-only, Python ast) for a syntax check only.
// Whether a reply is coherent, complete or invents turns is a human label (rubric A-E); automatic
// flags are hints for the reviewer, never labels. The predeclared rule (C1-C4) then picks the
// largest qualifying threshold, or keeps --eos-min-p opt-in.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "ef/chat.h"
#include "ef/tokenizer.h"

namespace ef {

class ThreadPool;

// One raw greedy trajectory: reply tokens (EOS excluded) and P(EOS) at every step
// (eos_p.size() == ids.size() + (eos ? 1 : 0)).
struct EosTrace {
  std::vector<std::int32_t> ids;
  std::vector<double> eos_p;
  bool eos = false;
  std::string source;  // "Hugging Face reference" / "BF16 engine" / "4-bit engine"
};

// Where a reply ends under `--eos-min-p tau` plus the chat guards.
struct EosCut {
  StopReason reason = StopReason::None;
  std::size_t kept = 0;       // generated tokens (as ChatSession would count them)
  std::string retained;       // the reply the user sees
  std::string removed;        // what the raw trajectory continues with after it
  bool borderline = false;    // some step's P(EOS) within 0.002 of tau before the stop
};

EosCut replay_eos_threshold(const Tokenizer& tok, const EosTrace& trace, double tau, const StopConfig& guards,
                            std::size_t max_new);

// Fenced code blocks (```lang ... ```); an unterminated last block runs to the end and is marked.
struct CodeBlock {
  std::string lang, code;
  bool closed = true;
};
std::vector<CodeBlock> extract_code_blocks(const std::string& text);

// Syntax check of the reply's code: "pass", "fail: <first error>", "no code block", "not checked: ...".
std::string check_code(const std::string& reply, const std::string& case_lang);

// One reply of one case at one threshold.
struct EosOutcome {
  std::string engine, case_name, kind, lang, system;
  std::size_t turn = 1;
  double tau = 0.0;
  std::string reason;  // stop_reason_name
  std::size_t kept = 0, raw_tokens = 0;
  bool raw_eos = false, borderline = false, marker_in_removed = false;
  std::string retained, removed, source, code, flags;
  std::string key;     // review key: same case, system, turn and retained text -> same key
};

struct EosLabelNeed {
  std::string key, case_name, kind, system, retained, removed, flags;
  std::size_t turn = 1;
  std::vector<std::string> engines;
  std::vector<double> taus;
};

// The predeclared decision for every threshold, per engine.
struct EosDecision {
  struct PerTau {
    double tau = 0.0;
    std::map<std::string, std::string> c1, c2, c3;  // engine -> "PASS ..." / "FAIL ..." / "PENDING ..."
    std::string verdict;                            // "qualifies" / "fails" / "pending labels"
  };
  std::vector<PerTau> taus;
  std::vector<EosLabelNeed> needed;  // replies that need a human label
  std::size_t missing = 0;           // ... of which have none yet
  std::string recommendation;
};

EosDecision decide_eos_threshold(const std::vector<EosOutcome>& outcomes, const std::vector<double>& taus,
                                 const std::map<std::string, std::string>& labels);

struct EosStudyOptions {
  std::filesystem::path golden;         // tests/golden/chat_eos_study.json (run mode)
  std::filesystem::path model_dir;      // BF16 checkpoint (+ tokenizer)
  std::filesystem::path qmodel;         // 4-bit engine (optional)
  std::filesystem::path tokenizer_dir;  // default: model_dir
  std::filesystem::path results;        // decide mode: an earlier ..._results.json instead of running
  std::filesystem::path labels;         // human labels (optional)
  std::string out = "bench/eos_study";  // writes <out>.md, <out>_results.json, <out>_labels_todo.json
  ThreadPool* pool = nullptr;
};

int run_eos_study(const EosStudyOptions& o);

}  // namespace ef
