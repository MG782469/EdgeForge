// EdgeForge - Stage 4: Llama/SentencePiece-style BPE tokenizer read from tokenizer.json.
//
// Encode (matches Hugging Face transformers for TinyLlama, legacy=false):
//   1. Split the raw text on added/special tokens ("<s>", "</s>", "<unk>"), longest match first.
//   2. In every other segment replace ' ' with U+2581 ("▁"). Prepend "▁" according to the
//      prepend scheme: "first" (only the segment that starts the text), "always", or "never";
//      nothing is prepended to a segment that already starts with "▁" (Metaspace rule).
//   3. BPE over the whole segment: start from single characters (characters missing from
//      the vocab become <0xXX> byte tokens), repeatedly merge the adjacent pair with the
//      lowest merge rank (leftmost on ties).
// Decode: tokens -> strings, "▁" -> ' ', runs of <0xXX> -> bytes (invalid UTF-8 -> U+FFFD per
// byte), concatenate, drop one leading space. Special tokens print verbatim or are skipped.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ef {

enum class PrependScheme { Always, First, Never };

class Tokenizer {
 public:
  // dir contains tokenizer.json and (optionally) tokenizer_config.json.
  static Tokenizer load(const std::filesystem::path& dir);
  static Tokenizer from_json(std::string_view tokenizer_json, std::string_view tokenizer_config_json = {});

  // add_bos: prepend <s>. parse_special: treat "<s>", "</s>", ... inside the text as special tokens
  // (Hugging Face default); false encodes them as ordinary text.
  std::vector<std::int32_t> encode(std::string_view text, bool add_bos = false, bool parse_special = true) const;
  std::string decode(const std::vector<std::int32_t>& ids, bool skip_special = false) const;

  const std::string& piece(std::int32_t id) const;  // raw vocabulary string, e.g. "▁Hello", "<0x0A>"
  bool is_special(std::int32_t id) const noexcept;
  std::size_t vocab_size() const noexcept { return pieces_.size(); }
  std::int32_t bos_id() const noexcept { return bos_id_; }
  std::int32_t eos_id() const noexcept { return eos_id_; }
  std::int32_t unk_id() const noexcept { return unk_id_; }
  const std::string& bos_token() const noexcept { return bos_token_; }
  const std::string& eos_token() const noexcept { return eos_token_; }
  const std::string& chat_template() const noexcept { return chat_template_; }
  PrependScheme prepend_scheme() const noexcept { return prepend_; }
  std::size_t merge_count() const noexcept { return merges_.size(); }

 private:
  void bpe(std::string_view word, std::vector<std::int32_t>& out) const;

  std::vector<std::string> pieces_;                    // id -> piece
  std::unordered_map<std::string, std::int32_t> ids_;  // piece -> id
  struct Merge {
    std::int32_t rank;
    std::int32_t id;
  };
  std::unordered_map<std::uint64_t, Merge> merges_;  // (left << 32 | right) -> rank, merged id
  std::int32_t byte_ids_[256];
  struct Added {
    std::string content;
    std::int32_t id;
    bool special;
  };
  std::vector<Added> added_;  // sorted longest first
  std::vector<bool> special_;
  PrependScheme prepend_ = PrependScheme::First;
  bool split_on_space_ = false;
  bool byte_fallback_ = true;
  bool fuse_unk_ = true;
  bool ignore_merges_ = false;
  std::int32_t unk_id_ = 0, bos_id_ = 1, eos_id_ = 2;
  std::string bos_token_ = "<s>", eos_token_ = "</s>", chat_template_;
};

// Incremental text output for streaming generation: feed token ids one by one and
// get back only the newly final text. A trailing run of <0xXX> byte tokens is held
// back until it ends, so the concatenated stream always equals decode() of the reply.
class StreamDecoder {
 public:
  explicit StreamDecoder(const Tokenizer& tok) : tok_(tok) {}
  std::string push(std::int32_t id);   // returns new text (may be empty)
  std::string flush();                 // anything still held back
  const std::string& text() const noexcept { return emitted_; }
  void reset() { ids_.clear(); emitted_.clear(); }

 private:
  std::string decode_ready(bool final) const;
  const Tokenizer& tok_;
  std::vector<std::int32_t> ids_;
  std::string emitted_;
};

// ---- chat template ---------------------------------------------------------

struct ChatMessage {
  std::string role;  // "system", "user" or "assistant"
  std::string content;
};

// TinyLlama-Chat / Zephyr template, byte-identical to tokenizer_config.json's Jinja:
//   <|system|>\n{content}</s>\n  <|user|>\n{content}</s>\n  <|assistant|>\n{content}</s>\n
//   + "<|assistant|>\n" when add_generation_prompt.
std::string apply_chat_template(const std::vector<ChatMessage>& messages, bool add_generation_prompt,
                                const std::string& eos_token = "</s>");

// True when `jinja` is the TinyLlama/Zephyr template implemented above.
bool is_zephyr_template(const std::string& jinja);

}  // namespace ef
