// EdgeForge - Stage 4 tokenizer implementation.
#include "ef/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <limits>

#include "ef/error.h"
#include "ef/json.h"

namespace ef {

namespace {

const std::string kSpace = "\xE2\x96\x81";       // U+2581 LOWER ONE EIGHTH BLOCK ("▁")
const std::string kReplacement = "\xEF\xBF\xBD";  // U+FFFD

std::string read_file(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw Error("cannot open '" + p.string() + "'");
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// Length of the well-formed UTF-8 character at s[i], or 0 if the bytes there are invalid.
std::size_t utf8_len(std::string_view s, std::size_t i) {
  const auto b0 = static_cast<unsigned char>(s[i]);
  auto cont = [&](std::size_t k, unsigned lo = 0x80, unsigned hi = 0xBF) {
    if (i + k >= s.size()) return false;
    const auto b = static_cast<unsigned char>(s[i + k]);
    return b >= lo && b <= hi;
  };
  if (b0 < 0x80) return 1;
  if (b0 >= 0xC2 && b0 <= 0xDF) return cont(1) ? 2 : 0;
  if (b0 == 0xE0) return cont(1, 0xA0) && cont(2) ? 3 : 0;
  if ((b0 >= 0xE1 && b0 <= 0xEC) || b0 == 0xEE || b0 == 0xEF) return cont(1) && cont(2) ? 3 : 0;
  if (b0 == 0xED) return cont(1, 0x80, 0x9F) && cont(2) ? 3 : 0;
  if (b0 == 0xF0) return cont(1, 0x90) && cont(2) && cont(3) ? 4 : 0;
  if (b0 >= 0xF1 && b0 <= 0xF3) return cont(1) && cont(2) && cont(3) ? 4 : 0;
  if (b0 == 0xF4) return cont(1, 0x80, 0x8F) && cont(2) && cont(3) ? 4 : 0;
  return 0;
}

bool valid_utf8(std::string_view s) {
  for (std::size_t i = 0; i < s.size();) {
    const std::size_t n = utf8_len(s, i);
    if (n == 0) return false;
    i += n;
  }
  return true;
}

// "<0xAB>" -> 0xAB, or -1.
int parse_byte_piece(const std::string& p) {
  if (p.size() != 6 || p[0] != '<' || p[1] != '0' || p[2] != 'x' || p[5] != '>') return -1;
  auto hex = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
  };
  const int hi = hex(p[3]), lo = hex(p[4]);
  return (hi < 0 || lo < 0) ? -1 : hi * 16 + lo;
}

std::string replace_all(std::string_view s, std::string_view from, std::string_view to) {
  std::string out;
  out.reserve(s.size());
  std::size_t i = 0;
  while (i < s.size()) {
    if (s.compare(i, from.size(), from) == 0) {
      out += to;
      i += from.size();
    } else {
      out += s[i++];
    }
  }
  return out;
}

// tokenizer_config.json tokens may be "…" or {"content": "…", ...}.
std::string token_string(const json::Value* v, const std::string& def) {
  if (v == nullptr || v->is_null()) return def;
  if (v->is_string()) return v->as_string();
  if (v->is_object() && v->find("content") && v->at("content").is_string()) return v->at("content").as_string();
  return def;
}

void scan_normalizer(const json::Value& n, bool& has_prepend, bool& has_replace) {
  if (!n.is_object()) return;
  const json::Value* type = n.find("type");
  if (type == nullptr || !type->is_string()) return;
  if (type->as_string() == "Sequence") {
    if (const json::Value* list = n.find("normalizers"); list && list->is_array()) {
      for (const auto& e : list->items()) scan_normalizer(e, has_prepend, has_replace);
    }
  } else if (type->as_string() == "Prepend") {
    has_prepend = true;
  } else if (type->as_string() == "Replace") {
    has_replace = true;
  }
}

}  // namespace

// ---------------------------------------------------------------- loading

Tokenizer Tokenizer::load(const std::filesystem::path& dir) {
  const auto tj = dir / "tokenizer.json";
  if (!std::filesystem::exists(tj)) {
    throw Error("tokenizer.json not found in '" + dir.string() + "' (it ships with the model download)");
  }
  const auto tc = dir / "tokenizer_config.json";
  const std::string cfg = std::filesystem::exists(tc) ? read_file(tc) : std::string();
  try {
    return from_json(read_file(tj), cfg);
  } catch (const Error& e) {
    throw Error("'" + tj.string() + "': " + e.what());
  }
}

Tokenizer Tokenizer::from_json(std::string_view tokenizer_json, std::string_view tokenizer_config_json) {
  Tokenizer t;
  std::fill(std::begin(t.byte_ids_), std::end(t.byte_ids_), -1);
  const json::Value root = json::parse(tokenizer_json);
  const json::Value& model = root.at("model");
  if (const json::Value* type = model.find("type"); type && type->is_string() && type->as_string() != "BPE") {
    throw Error("tokenizer model type '" + type->as_string() + "' is not supported (expected BPE)");
  }

  // vocab
  const json::Value& vocab = model.at("vocab");
  std::int64_t max_id = -1;
  for (const auto& v : vocab.items()) max_id = std::max(max_id, v.as_int64());
  if (const json::Value* added = root.find("added_tokens"); added && added->is_array()) {
    for (const auto& a : added->items()) max_id = std::max(max_id, a.at("id").as_int64());
  }
  if (max_id < 0 || max_id > (1 << 24)) throw Error("tokenizer vocab is empty or too large");
  t.pieces_.assign(static_cast<std::size_t>(max_id + 1), std::string());
  t.special_.assign(t.pieces_.size(), false);
  for (std::size_t i = 0; i < vocab.keys().size(); ++i) {
    const auto id = static_cast<std::int32_t>(vocab.items()[i].as_int64());
    if (id < 0) throw Error("negative token id in vocab");
    t.pieces_[static_cast<std::size_t>(id)] = vocab.keys()[i];
    t.ids_[vocab.keys()[i]] = id;
    const int b = parse_byte_piece(vocab.keys()[i]);
    if (b >= 0) t.byte_ids_[b] = id;
  }

  // merges: "a b" strings (classic) or ["a", "b"] pairs (tokenizers >= 0.20)
  const json::Value& merges = model.at("merges");
  std::int32_t rank = 0;
  std::size_t skipped = 0;
  for (const auto& m : merges.items()) {
    std::string a, b;
    if (m.is_string()) {
      const std::string& s = m.as_string();
      const auto sp = s.find(' ');
      if (sp == std::string::npos) throw Error("malformed merge '" + s + "'");
      a = s.substr(0, sp);
      b = s.substr(sp + 1);
    } else {
      a = m[0].as_string();
      b = m[1].as_string();
    }
    const auto ia = t.ids_.find(a), ib = t.ids_.find(b), im = t.ids_.find(a + b);
    if (ia == t.ids_.end() || ib == t.ids_.end() || im == t.ids_.end()) {
      ++skipped;
      ++rank;
      continue;
    }
    const std::uint64_t key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(ia->second)) << 32) |
                              static_cast<std::uint32_t>(ib->second);
    t.merges_.emplace(key, Merge{rank++, im->second});  // first occurrence keeps the best rank
  }
  if (skipped > 0 && t.merges_.empty()) throw Error("no usable merges in tokenizer.json");

  auto get_bool = [](const json::Value& o, const char* k, bool def) {
    const json::Value* v = o.find(k);
    return (v && v->is_bool()) ? v->as_bool() : def;
  };
  t.byte_fallback_ = get_bool(model, "byte_fallback", false);
  t.fuse_unk_ = get_bool(model, "fuse_unk", false);
  t.ignore_merges_ = get_bool(model, "ignore_merges", false);
  if (const json::Value* unk = model.find("unk_token"); unk && unk->is_string()) {
    const auto it = t.ids_.find(unk->as_string());
    if (it != t.ids_.end()) t.unk_id_ = it->second;
  }

  // added tokens (matched in raw text)
  if (const json::Value* added = root.find("added_tokens"); added && added->is_array()) {
    for (const auto& a : added->items()) {
      Added at;
      at.id = static_cast<std::int32_t>(a.at("id").as_int64());
      at.content = a.at("content").as_string();
      at.special = get_bool(a, "special", false);
      if (at.content.empty()) continue;
      t.pieces_[static_cast<std::size_t>(at.id)] = at.content;
      t.ids_.emplace(at.content, at.id);
      t.special_[static_cast<std::size_t>(at.id)] = at.special;
      t.added_.push_back(at);
    }
  }
  std::stable_sort(t.added_.begin(), t.added_.end(),
                   [](const Added& x, const Added& y) { return x.content.size() > y.content.size(); });

  // Space handling. tokenizer.json may use the legacy normalizer (Prepend + Replace) or a
  // Metaspace pre-tokenizer; transformers with legacy=false behaves like Metaspace "first".
  bool has_prepend = false, has_replace = false;
  if (const json::Value* n = root.find("normalizer"); n) scan_normalizer(*n, has_prepend, has_replace);
  bool metaspace = false;
  if (const json::Value* p = root.find("pre_tokenizer"); p && p->is_object()) {
    const json::Value* type = p->find("type");
    if (type && type->is_string() && type->as_string() == "Metaspace") {
      metaspace = true;
      t.split_on_space_ = get_bool(*p, "split", false);
      if (const json::Value* sch = p->find("prepend_scheme"); sch && sch->is_string()) {
        const std::string s = sch->as_string();
        t.prepend_ = s == "always" ? PrependScheme::Always : s == "never" ? PrependScheme::Never : PrependScheme::First;
      } else {
        t.prepend_ = get_bool(*p, "add_prefix_space", true) ? PrependScheme::Always : PrependScheme::Never;
      }
    }
  }
  if (!metaspace) t.prepend_ = has_prepend ? PrependScheme::Always : PrependScheme::Never;

  // tokenizer_config.json: special token strings, legacy flag, chat template
  if (!tokenizer_config_json.empty()) {
    const json::Value cfg = json::parse(tokenizer_config_json);
    t.bos_token_ = token_string(cfg.find("bos_token"), t.bos_token_);
    t.eos_token_ = token_string(cfg.find("eos_token"), t.eos_token_);
    if (const json::Value* ct = cfg.find("chat_template"); ct && ct->is_string()) t.chat_template_ = ct->as_string();
    if (const json::Value* legacy = cfg.find("legacy"); legacy && legacy->is_bool() && !legacy->as_bool()) {
      t.prepend_ = PrependScheme::First;  // transformers: legacy=false -> no "▁" after special tokens
    }
  }
  if (auto it = t.ids_.find(t.bos_token_); it != t.ids_.end()) t.bos_id_ = it->second;
  if (auto it = t.ids_.find(t.eos_token_); it != t.ids_.end()) t.eos_id_ = it->second;
  return t;
}

// ---------------------------------------------------------------- encode

void Tokenizer::bpe(std::string_view word, std::vector<std::int32_t>& out) const {
  if (word.empty()) return;
  if (ignore_merges_) {
    if (const auto it = ids_.find(std::string(word)); it != ids_.end()) {
      out.push_back(it->second);
      return;
    }
  }
  // Initial symbols: one per character; unknown characters -> byte tokens (or <unk>).
  std::vector<std::int32_t> sym;
  sym.reserve(word.size());
  bool last_unk = false;
  for (std::size_t i = 0; i < word.size();) {
    std::size_t n = utf8_len(word, i);
    if (n == 0) n = 1;  // invalid byte: handled as a lone byte below
    const auto it = ids_.find(std::string(word.substr(i, n)));
    if (it != ids_.end() && !(n == 1 && static_cast<unsigned char>(word[i]) >= 0x80)) {
      sym.push_back(it->second);
      last_unk = false;
    } else if (byte_fallback_ && byte_ids_[static_cast<unsigned char>(word[i])] >= 0) {
      for (std::size_t k = 0; k < n; ++k) sym.push_back(byte_ids_[static_cast<unsigned char>(word[i + k])]);
      last_unk = false;
    } else {
      if (!(fuse_unk_ && last_unk)) sym.push_back(unk_id_);
      last_unk = true;
    }
    i += n;
  }
  // Merge the lowest-rank adjacent pair (leftmost on ties) until none applies.
  while (sym.size() > 1) {
    std::int32_t best_rank = std::numeric_limits<std::int32_t>::max();
    std::size_t best_pos = 0;
    std::int32_t best_id = -1;
    for (std::size_t i = 0; i + 1 < sym.size(); ++i) {
      const std::uint64_t key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(sym[i])) << 32) |
                                static_cast<std::uint32_t>(sym[i + 1]);
      const auto it = merges_.find(key);
      if (it != merges_.end() && it->second.rank < best_rank) {
        best_rank = it->second.rank;
        best_pos = i;
        best_id = it->second.id;
      }
    }
    if (best_id < 0) break;
    sym[best_pos] = best_id;
    sym.erase(sym.begin() + static_cast<std::ptrdiff_t>(best_pos) + 1);
  }
  out.insert(out.end(), sym.begin(), sym.end());
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view text, bool add_bos, bool parse_special) const {
  std::vector<std::int32_t> out;
  if (add_bos) out.push_back(bos_id_);
  auto encode_segment = [&](std::string_view seg, std::size_t offset) {
    if (seg.empty()) return;
    std::string s = replace_all(seg, " ", kSpace);
    const bool starts_with_space = s.compare(0, kSpace.size(), kSpace) == 0;
    if (prepend_ == PrependScheme::Always || (prepend_ == PrependScheme::First && offset == 0)) {
      if (!starts_with_space) s = kSpace + s;
    }
    if (!split_on_space_) {
      bpe(s, out);
      return;
    }
    // Metaspace split=true: each "▁" starts a new word (MergedWithNext).
    std::size_t start = 0;
    for (std::size_t i = kSpace.size(); i <= s.size(); ++i) {
      if (i == s.size() || s.compare(i, kSpace.size(), kSpace) == 0) {
        bpe(std::string_view(s).substr(start, i - start), out);
        start = i;
      }
    }
  };
  std::size_t seg_start = 0, i = 0;
  while (i < text.size()) {
    const Added* hit = nullptr;
    if (parse_special) {
      for (const Added& a : added_) {
        if (text.compare(i, a.content.size(), a.content) == 0) {
          hit = &a;
          break;
        }
      }
    }
    if (hit == nullptr) {
      ++i;
      continue;
    }
    encode_segment(text.substr(seg_start, i - seg_start), seg_start);
    out.push_back(hit->id);
    i += hit->content.size();
    seg_start = i;
  }
  encode_segment(text.substr(seg_start), seg_start);
  return out;
}

// ---------------------------------------------------------------- decode

const std::string& Tokenizer::piece(std::int32_t id) const {
  if (id < 0 || static_cast<std::size_t>(id) >= pieces_.size()) {
    throw Error("token id " + std::to_string(id) + " is outside the vocabulary (" + std::to_string(pieces_.size()) + ")");
  }
  return pieces_[static_cast<std::size_t>(id)];
}

bool Tokenizer::is_special(std::int32_t id) const noexcept {
  return id >= 0 && static_cast<std::size_t>(id) < special_.size() && special_[static_cast<std::size_t>(id)];
}

std::string Tokenizer::decode(const std::vector<std::int32_t>& ids, bool skip_special) const {
  std::string out;
  std::string bytes;  // pending run of <0xXX> tokens
  std::size_t byte_tokens = 0;
  auto flush_bytes = [&]() {
    if (byte_tokens == 0) return;
    if (valid_utf8(bytes)) {
      out += bytes;
    } else {
      for (std::size_t k = 0; k < byte_tokens; ++k) out += kReplacement;
    }
    bytes.clear();
    byte_tokens = 0;
  };
  for (const std::int32_t id : ids) {
    if (skip_special && is_special(id)) continue;
    const std::string& p = piece(id);
    const int b = is_special(id) ? -1 : parse_byte_piece(p);
    if (b >= 0) {
      bytes += static_cast<char>(b);
      ++byte_tokens;
      continue;
    }
    flush_bytes();
    out += replace_all(p, kSpace, " ");
  }
  flush_bytes();
  if (!out.empty() && out.front() == ' ') out.erase(0, 1);  // Strip(' ', start=1)
  return out;
}

std::string StreamDecoder::decode_ready(bool final) const {
  // A run of <0xXX> byte tokens is decoded as one unit (valid UTF-8 -> text, otherwise one
  // U+FFFD per byte, exactly like Hugging Face), so a run can change meaning until it ends.
  // Hold back the trailing byte run until a normal token (or the end of the reply) closes it:
  // everything emitted is then final and the stream equals decode() of the whole reply.
  std::size_t n = ids_.size();
  if (!final) {
    while (n > 0 && !tok_.is_special(ids_[n - 1]) && parse_byte_piece(tok_.piece(ids_[n - 1])) >= 0) --n;
  }
  return tok_.decode(std::vector<std::int32_t>(ids_.begin(), ids_.begin() + static_cast<std::ptrdiff_t>(n)), true);
}

std::string StreamDecoder::push(std::int32_t id) {
  ids_.push_back(id);
  const std::string now = decode_ready(false);
  if (now.size() < emitted_.size() || now.compare(0, emitted_.size(), emitted_) != 0) {
    // Should not happen (decoding is prefix-stable once bytes are complete); resync.
    emitted_ = now;
    return {};
  }
  std::string delta = now.substr(emitted_.size());
  emitted_ = now;
  return delta;
}

std::string StreamDecoder::flush() {
  const std::string now = decode_ready(true);
  std::string delta = now.size() > emitted_.size() ? now.substr(emitted_.size()) : std::string();
  emitted_ = now;
  return delta;
}

// ---------------------------------------------------------------- chat template

std::string apply_chat_template(const std::vector<ChatMessage>& messages, bool add_generation_prompt,
                                const std::string& eos_token) {
  std::string out;
  for (const auto& m : messages) {
    if (m.role == "user" || m.role == "system" || m.role == "assistant") {
      out += "<|" + m.role + "|>\n" + m.content + eos_token + "\n";
    } else {
      throw Error("chat template: unknown role '" + m.role + "'");
    }
  }
  if (add_generation_prompt && !messages.empty()) out += "<|assistant|>\n";
  return out;
}

bool is_zephyr_template(const std::string& jinja) {
  static const std::string kZephyr =
      "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + "
      "eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token "
      "}}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% "
      "endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}";
  auto squeeze = [](const std::string& s) {
    std::string r;
    for (char c : s) {
      if (!std::isspace(static_cast<unsigned char>(c))) r += c;
    }
    return r;
  };
  return squeeze(jinja) == squeeze(kZephyr);
}

}  // namespace ef
