// EdgeForge — minimal, strict JSON parser (RFC 8259).
//
// Used for config.json, the safetensors header and (in Stage 4) tokenizer.json.
// Strict by design: duplicate object keys, trailing commas, leading zeros,
// invalid UTF-8, unpaired surrogates, raw control characters, NaN/Infinity and
// trailing garbage are all rejected with a line/column error message.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ef/error.h"

namespace ef::json {

enum class Type { Null, Bool, Number, String, Array, Object };
const char* type_name(Type t) noexcept;

struct ParseOptions {
  std::size_t max_depth = 256;        // nesting limit (protects the stack)
  bool allow_duplicate_keys = false;  // RFC says SHOULD be unique; we require it
};

class Value {
 public:
  Value() = default;

  Type type() const noexcept { return type_; }
  bool is_null() const noexcept { return type_ == Type::Null; }
  bool is_bool() const noexcept { return type_ == Type::Bool; }
  bool is_number() const noexcept { return type_ == Type::Number; }
  bool is_integer() const noexcept { return type_ == Type::Number && is_int_; }
  bool is_string() const noexcept { return type_ == Type::String; }
  bool is_array() const noexcept { return type_ == Type::Array; }
  bool is_object() const noexcept { return type_ == Type::Object; }

  // Typed accessors: throw ef::Error on a type mismatch.
  bool as_bool() const;
  double as_double() const;
  std::int64_t as_int64() const;    // integral literals only (no '.', no exponent)
  std::uint64_t as_uint64() const;  // non-negative integral literals only
  const std::string& as_string() const;
  const std::string& number_text() const;  // exact literal text of a number

  // Arrays and objects.
  std::size_t size() const;                       // element / member count
  const Value& operator[](std::size_t i) const;   // array element, bounds-checked
  const std::vector<Value>& items() const;        // array elements or object values
  const std::vector<std::string>& keys() const;   // object keys in document order
  const Value* find(std::string_view key) const;  // object member or nullptr
  const Value& at(std::string_view key) const;    // object member, throws if missing

 private:
  friend class Parser;
  void expect(Type t) const;

  Type type_ = Type::Null;
  bool bool_ = false;
  bool is_int_ = false;
  std::int64_t int_ = 0;
  double num_ = 0.0;
  std::string str_;                // string payload, or literal text of a number
  std::vector<Value> items_;       // array elements / object values
  std::vector<std::string> keys_;  // object keys (parallel to items_)
  std::shared_ptr<const std::unordered_map<std::string, std::size_t>> index_;
};

// Parses a complete JSON document. Throws ef::Error with line/column on failure.
Value parse(std::string_view text, const ParseOptions& options = {});

// Escapes a UTF-8 string for use inside a JSON string literal (no quotes added).
std::string escape(std::string_view s);

}  // namespace ef::json
