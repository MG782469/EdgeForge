// EdgeForge — strict JSON parser implementation.
#include "ef/json.h"

#include <charconv>
#include <cmath>
#include <locale>
#include <sstream>
#include <system_error>
#include <version>

namespace ef::json {

const char* type_name(Type t) noexcept {
  switch (t) {
    case Type::Null: return "null";
    case Type::Bool: return "bool";
    case Type::Number: return "number";
    case Type::String: return "string";
    case Type::Array: return "array";
    case Type::Object: return "object";
  }
  return "unknown";
}

// ---------------------------------------------------------------- Value

void Value::expect(Type t) const {
  if (type_ != t) {
    throw Error(std::string("JSON: expected ") + type_name(t) + ", got " + type_name(type_));
  }
}

bool Value::as_bool() const {
  expect(Type::Bool);
  return bool_;
}

double Value::as_double() const {
  expect(Type::Number);
  return num_;
}

std::int64_t Value::as_int64() const {
  expect(Type::Number);
  if (!is_int_) throw Error("JSON: expected an integer, got " + str_);
  return int_;
}

std::uint64_t Value::as_uint64() const {
  const std::int64_t v = as_int64();
  if (v < 0) throw Error("JSON: expected a non-negative integer, got " + str_);
  return static_cast<std::uint64_t>(v);
}

const std::string& Value::as_string() const {
  expect(Type::String);
  return str_;
}

const std::string& Value::number_text() const {
  expect(Type::Number);
  return str_;
}

std::size_t Value::size() const {
  if (type_ != Type::Array && type_ != Type::Object) {
    throw Error(std::string("JSON: size() needs an array or object, got ") + type_name(type_));
  }
  return items_.size();
}

const Value& Value::operator[](std::size_t i) const {
  expect(Type::Array);
  if (i >= items_.size()) {
    throw Error("JSON: array index " + std::to_string(i) + " out of range (size " +
                std::to_string(items_.size()) + ")");
  }
  return items_[i];
}

const std::vector<Value>& Value::items() const {
  if (type_ != Type::Array && type_ != Type::Object) {
    throw Error(std::string("JSON: items() needs an array or object, got ") + type_name(type_));
  }
  return items_;
}

const std::vector<std::string>& Value::keys() const {
  expect(Type::Object);
  return keys_;
}

const Value* Value::find(std::string_view key) const {
  expect(Type::Object);
  if (index_) {
    auto it = index_->find(std::string(key));
    return it == index_->end() ? nullptr : &items_[it->second];
  }
  for (std::size_t i = 0; i < keys_.size(); ++i) {
    if (keys_[i] == key) return &items_[i];
  }
  return nullptr;
}

const Value& Value::at(std::string_view key) const {
  const Value* v = find(key);
  if (v == nullptr) throw Error("JSON: missing key \"" + std::string(key) + "\"");
  return *v;
}

// ---------------------------------------------------------------- helpers

namespace {

bool is_digit(char c) { return c >= '0' && c <= '9'; }

std::string describe_char(char c) {
  const auto u = static_cast<unsigned char>(c);
  if (u >= 0x20 && u < 0x7f) return std::string("'") + c + "'";
  static const char* hex = "0123456789ABCDEF";
  return std::string("byte 0x") + hex[u >> 4] + hex[u & 0xF];
}

std::string clip(std::string_view s, std::size_t max_len = 64) {
  if (s.size() <= max_len) return std::string(s);
  return std::string(s.substr(0, max_len)) + "...";
}

void append_utf8(std::string& out, std::uint32_t cp) {
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

// Returns the length of a valid UTF-8 sequence starting at s[i], or 0 if the
// bytes are not well-formed UTF-8 (RFC 3629: no overlongs, no surrogates,
// nothing above U+10FFFF).
std::size_t utf8_sequence_length(std::string_view s, std::size_t i) {
  auto byte = [&](std::size_t k) -> int {
    return k < s.size() ? static_cast<unsigned char>(s[k]) : -1;
  };
  auto cont = [&](std::size_t k, int lo = 0x80, int hi = 0xBF) {
    const int b = byte(k);
    return b >= lo && b <= hi;
  };
  const int b0 = byte(i);
  if (b0 >= 0xC2 && b0 <= 0xDF) return cont(i + 1) ? 2 : 0;
  if (b0 == 0xE0) return cont(i + 1, 0xA0, 0xBF) && cont(i + 2) ? 3 : 0;
  if ((b0 >= 0xE1 && b0 <= 0xEC) || b0 == 0xEE || b0 == 0xEF) return cont(i + 1) && cont(i + 2) ? 3 : 0;
  if (b0 == 0xED) return cont(i + 1, 0x80, 0x9F) && cont(i + 2) ? 3 : 0;
  if (b0 == 0xF0) return cont(i + 1, 0x90, 0xBF) && cont(i + 2) && cont(i + 3) ? 4 : 0;
  if (b0 >= 0xF1 && b0 <= 0xF3) return cont(i + 1) && cont(i + 2) && cont(i + 3) ? 4 : 0;
  if (b0 == 0xF4) return cont(i + 1, 0x80, 0x8F) && cont(i + 2) && cont(i + 3) ? 4 : 0;
  return 0;
}

// Locale-independent decimal-to-double conversion. Returns false on overflow.
bool to_double(std::string_view text, double& out) {
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
  const char* b = text.data();
  const char* e = text.data() + text.size();
  const auto r = std::from_chars(b, e, out);
  if (r.ec == std::errc() && r.ptr == e) return true;
  if (r.ec != std::errc::result_out_of_range) return false;
  // Out of range: distinguish underflow (accept, value ~0) from overflow (reject).
#endif
  std::istringstream is{std::string(text)};
  is.imbue(std::locale::classic());
  double d = 0.0;
  is >> d;
  if (std::isfinite(d) && std::fabs(d) < 1.0) {  // underflow to zero/denormal
    out = d;
    return true;
  }
  if (!is.fail() && std::isfinite(d)) {
    out = d;
    return true;
  }
  return false;
}

}  // namespace

// ---------------------------------------------------------------- Parser

class Parser {
 public:
  Parser(std::string_view text, const ParseOptions& options) : s_(text), opt_(options) {}

  Value run() {
    skip_ws();
    Value v = parse_value(0);
    skip_ws();
    if (i_ != s_.size()) fail("unexpected trailing characters after the JSON value");
    return v;
  }

 private:
  std::string_view s_;
  const ParseOptions& opt_;
  std::size_t i_ = 0;

  [[noreturn]] void fail(const std::string& msg) const { fail_at(i_, msg); }

  [[noreturn]] void fail_at(std::size_t pos, const std::string& msg) const {
    std::size_t line = 1;
    std::size_t col = 1;
    for (std::size_t k = 0; k < pos && k < s_.size(); ++k) {
      if (s_[k] == '\n') {
        ++line;
        col = 1;
      } else {
        ++col;
      }
    }
    throw Error("JSON parse error at line " + std::to_string(line) + ", column " +
                std::to_string(col) + " (byte " + std::to_string(pos) + "): " + msg);
  }

  bool eof() const { return i_ >= s_.size(); }

  void skip_ws() {
    while (!eof()) {
      const char c = s_[i_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++i_;
      } else {
        break;
      }
    }
  }

  Value parse_value(std::size_t depth) {
    if (eof()) fail("unexpected end of input, expected a value");
    const char c = s_[i_];
    switch (c) {
      case '{': return parse_object(depth + 1);
      case '[': return parse_array(depth + 1);
      case '"': {
        Value v;
        v.type_ = Type::String;
        v.str_ = parse_string();
        return v;
      }
      case 't': return parse_literal("true");
      case 'f': return parse_literal("false");
      case 'n': return parse_literal("null");
      default: break;
    }
    if (c == '-' || is_digit(c)) return parse_number();
    fail("unexpected character " + describe_char(c) + ", expected a value");
  }

  Value parse_literal(std::string_view word) {
    if (s_.substr(i_, word.size()) != word) {
      fail("invalid literal (expected '" + std::string(word) + "')");
    }
    i_ += word.size();
    Value v;
    if (word == "null") {
      v.type_ = Type::Null;
    } else {
      v.type_ = Type::Bool;
      v.bool_ = (word == "true");
    }
    return v;
  }

  void check_depth(std::size_t depth) const {
    if (depth > opt_.max_depth) {
      fail("nesting depth exceeds the limit of " + std::to_string(opt_.max_depth));
    }
  }

  Value parse_object(std::size_t depth) {
    check_depth(depth);
    ++i_;  // '{'
    Value obj;
    obj.type_ = Type::Object;
    auto index = std::make_shared<std::unordered_map<std::string, std::size_t>>();
    skip_ws();
    if (!eof() && s_[i_] == '}') {
      ++i_;
      obj.index_ = std::move(index);
      return obj;
    }
    while (true) {
      skip_ws();
      if (eof()) fail("unexpected end of input inside an object");
      if (s_[i_] != '"') fail("expected a string key in object, got " + describe_char(s_[i_]));
      const std::size_t key_pos = i_;
      std::string key = parse_string();
      skip_ws();
      if (eof() || s_[i_] != ':') fail("expected ':' after object key \"" + clip(key) + "\"");
      ++i_;
      skip_ws();
      Value v = parse_value(depth);
      const auto [it, inserted] = index->emplace(key, obj.items_.size());
      if (!inserted) {
        if (!opt_.allow_duplicate_keys) fail_at(key_pos, "duplicate key \"" + clip(key) + "\"");
        obj.items_[it->second] = std::move(v);  // last one wins when allowed
      } else {
        obj.keys_.push_back(std::move(key));
        obj.items_.push_back(std::move(v));
      }
      skip_ws();
      if (eof()) fail("unexpected end of input inside an object");
      const char c = s_[i_];
      if (c == ',') {
        ++i_;
        skip_ws();
        if (!eof() && s_[i_] == '}') fail("trailing comma in object");
        continue;
      }
      if (c == '}') {
        ++i_;
        break;
      }
      fail("expected ',' or '}' in object, got " + describe_char(c));
    }
    obj.index_ = std::move(index);
    return obj;
  }

  Value parse_array(std::size_t depth) {
    check_depth(depth);
    ++i_;  // '['
    Value arr;
    arr.type_ = Type::Array;
    skip_ws();
    if (!eof() && s_[i_] == ']') {
      ++i_;
      return arr;
    }
    while (true) {
      skip_ws();
      arr.items_.push_back(parse_value(depth));
      skip_ws();
      if (eof()) fail("unexpected end of input inside an array");
      const char c = s_[i_];
      if (c == ',') {
        ++i_;
        skip_ws();
        if (!eof() && s_[i_] == ']') fail("trailing comma in array");
        continue;
      }
      if (c == ']') {
        ++i_;
        break;
      }
      fail("expected ',' or ']' in array, got " + describe_char(c));
    }
    return arr;
  }

  std::uint32_t parse_hex4() {
    if (i_ + 4 > s_.size()) fail("invalid \\u escape: expected 4 hex digits");
    std::uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
      const char c = s_[i_++];
      v <<= 4;
      if (c >= '0' && c <= '9') {
        v |= static_cast<std::uint32_t>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        v |= static_cast<std::uint32_t>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        v |= static_cast<std::uint32_t>(c - 'A' + 10);
      } else {
        fail_at(i_ - 1, "invalid \\u escape: " + describe_char(c) + " is not a hex digit");
      }
    }
    return v;
  }

  std::string parse_string() {
    const std::size_t start = i_;
    ++i_;  // opening quote
    std::string out;
    while (true) {
      if (eof()) fail_at(start, "unterminated string");
      const auto c = static_cast<unsigned char>(s_[i_]);
      if (c == '"') {
        ++i_;
        return out;
      }
      if (c == '\\') {
        const std::size_t esc_pos = i_;
        ++i_;
        if (eof()) fail_at(start, "unterminated string");
        const char e = s_[i_++];
        switch (e) {
          case '"': out += '"'; break;
          case '\\': out += '\\'; break;
          case '/': out += '/'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'n': out += '\n'; break;
          case 'r': out += '\r'; break;
          case 't': out += '\t'; break;
          case 'u': {
            std::uint32_t cp = parse_hex4();
            if (cp >= 0xD800 && cp <= 0xDBFF) {
              if (i_ + 1 < s_.size() && s_[i_] == '\\' && s_[i_ + 1] == 'u') {
                i_ += 2;
                const std::uint32_t lo = parse_hex4();
                if (lo < 0xDC00 || lo > 0xDFFF) {
                  fail_at(esc_pos, "high surrogate is not followed by a low surrogate");
                }
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              } else {
                fail_at(esc_pos, "unpaired high surrogate in \\u escape");
              }
            } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
              fail_at(esc_pos, "unpaired low surrogate in \\u escape");
            }
            append_utf8(out, cp);
            break;
          }
          default:
            fail_at(esc_pos, "invalid escape sequence \\" + std::string(1, e));
        }
        continue;
      }
      if (c < 0x20) fail("unescaped control character " + describe_char(static_cast<char>(c)) + " in string");
      if (c < 0x80) {
        out += static_cast<char>(c);
        ++i_;
        continue;
      }
      const std::size_t len = utf8_sequence_length(s_, i_);
      if (len == 0) fail("invalid UTF-8 byte sequence in string");
      out.append(s_.data() + i_, len);
      i_ += len;
    }
  }

  Value parse_number() {
    const std::size_t start = i_;
    if (s_[i_] == '-') ++i_;
    if (eof()) fail_at(start, "invalid number: missing digits");
    if (s_[i_] == '0') {
      ++i_;
      if (!eof() && is_digit(s_[i_])) fail_at(start, "invalid number: leading zeros are not allowed");
    } else if (is_digit(s_[i_])) {
      while (!eof() && is_digit(s_[i_])) ++i_;
    } else {
      fail_at(start, "invalid number: expected a digit");
    }
    bool integral = true;
    if (!eof() && s_[i_] == '.') {
      ++i_;
      integral = false;
      if (eof() || !is_digit(s_[i_])) fail_at(start, "invalid number: expected a digit after '.'");
      while (!eof() && is_digit(s_[i_])) ++i_;
    }
    if (!eof() && (s_[i_] == 'e' || s_[i_] == 'E')) {
      ++i_;
      integral = false;
      if (!eof() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
      if (eof() || !is_digit(s_[i_])) fail_at(start, "invalid number: expected a digit in the exponent");
      while (!eof() && is_digit(s_[i_])) ++i_;
    }
    const std::string_view text = s_.substr(start, i_ - start);
    Value v;
    v.type_ = Type::Number;
    v.str_ = std::string(text);
    double d = 0.0;
    if (!to_double(text, d)) fail_at(start, "number out of range: " + clip(text));
    v.num_ = d;
    if (integral) {
      std::int64_t iv = 0;
      const auto r = std::from_chars(text.data(), text.data() + text.size(), iv);
      if (r.ec == std::errc() && r.ptr == text.data() + text.size()) {
        v.is_int_ = true;
        v.int_ = iv;
      }
    }
    return v;
  }
};

Value parse(std::string_view text, const ParseOptions& options) {
  Parser p(text, options);
  return p.run();
}

std::string escape(std::string_view s) {
  static const char* hex = "0123456789abcdef";
  std::string out;
  out.reserve(s.size() + 8);
  for (const char ch : s) {
    const auto c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          out += "\\u00";
          out += hex[c >> 4];
          out += hex[c & 0xF];
        } else {
          out += ch;
        }
    }
  }
  return out;
}

}  // namespace ef::json
