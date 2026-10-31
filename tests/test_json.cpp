// EdgeForge — JSON parser tests (valid documents, every rejection path, fuzz).
#include <random>
#include <string>

#include "ef/json.h"
#include "test_framework.h"

using ef::json::parse;
using ef::json::Type;

TEST(json_primitives) {
  CHECK(parse("null").is_null());
  CHECK_EQ(parse("true").as_bool(), true);
  CHECK_EQ(parse("false").as_bool(), false);
  CHECK_EQ(parse("0").as_int64(), 0);
  CHECK_EQ(parse("-0").as_int64(), 0);
  CHECK_EQ(parse("123").as_int64(), 123);
  CHECK_EQ(parse("-45").as_int64(), -45);
  CHECK_EQ(parse("1.5").as_double(), 1.5);
  CHECK_EQ(parse("1e3").as_double(), 1000.0);
  CHECK_EQ(parse("1E-2").as_double(), 0.01);
  CHECK_EQ(parse("1e-05").as_double(), 1e-05);
  CHECK(!parse("1.0").is_integer());  // has a fraction part, not an integer literal
  CHECK(!parse("1e3").is_integer());
  CHECK_EQ(parse("9223372036854775807").as_int64(), INT64_C(9223372036854775807));
  CHECK(!parse("9223372036854775808").is_integer());  // beyond int64: kept as double
  CHECK_EQ(parse("9223372036854775808").as_double(), 9223372036854775808.0);
  CHECK_EQ(parse("2200096768").as_uint64(), UINT64_C(2200096768));
  CHECK_EQ(parse("1e-400").as_double(), 0.0);  // underflow is accepted as 0
  CHECK_EQ(parse("  \t\r\n 42 \n").as_int64(), 42);
  CHECK_EQ(parse("12.5").number_text(), std::string("12.5"));
}

TEST(json_strings_and_escapes) {
  CHECK_EQ(parse(R"("hello")").as_string(), std::string("hello"));
  CHECK_EQ(parse(R"("a\"b\\c\/d")").as_string(), std::string("a\"b\\c/d"));
  CHECK_EQ(parse(R"("\b\f\n\r\t")").as_string(), std::string("\b\f\n\r\t"));
  CHECK_EQ(parse("\"\\u0041\"").as_string(), std::string("A"));
  CHECK_EQ(parse("\"\\u00e9\"").as_string(), std::string("\xC3\xA9"));            // e-acute
  CHECK_EQ(parse("\"\\u2581\"").as_string(), std::string("\xE2\x96\x81"));        // SentencePiece space
  CHECK_EQ(parse("\"\\uD83D\\uDE00\"").as_string(), std::string("\xF0\x9F\x98\x80"));  // surrogate pair -> emoji
  CHECK_EQ(parse("\"\\u0000\"").as_string().size(), std::size_t{1});              // embedded NUL kept
  CHECK_EQ(parse("\"नमस्ते\"").as_string(), std::string("नमस्ते"));             // raw UTF-8 passes through
  CHECK_EQ(parse("\"\xF0\x9F\x98\x80\"").as_string(), std::string("\xF0\x9F\x98\x80"));
}

TEST(json_objects_and_arrays) {
  const auto v = parse(R"({"b": 1, "a": [true, null, "x", {"k": -2.5}], "c": {}})");
  CHECK(v.is_object());
  CHECK_EQ(v.size(), std::size_t{3});
  CHECK_EQ(v.keys()[0], std::string("b"));  // document order preserved
  CHECK_EQ(v.keys()[1], std::string("a"));
  CHECK_EQ(v.at("b").as_int64(), 1);
  CHECK(v.find("missing") == nullptr);
  const auto& a = v.at("a");
  CHECK_EQ(a.size(), std::size_t{4});
  CHECK(a[1].is_null());
  CHECK_EQ(a[2].as_string(), std::string("x"));
  CHECK_EQ(a[3].at("k").as_double(), -2.5);
  CHECK_EQ(v.at("c").size(), std::size_t{0});
  CHECK_EQ(parse("[]").size(), std::size_t{0});
  CHECK_THROWS(v.at("missing"), "missing key");
  CHECK_THROWS(a[9], "out of range");
}

TEST(json_type_errors) {
  CHECK_THROWS(parse("1").as_string(), "expected string");
  CHECK_THROWS(parse("\"x\"").as_int64(), "expected number");
  CHECK_THROWS(parse("1.5").as_int64(), "expected an integer");
  CHECK_THROWS(parse("-3").as_uint64(), "non-negative");
  CHECK_THROWS(parse("[1]").at("k"), "expected object");
  CHECK_THROWS(parse("true").size(), "array or object");
}

TEST(json_rejects_malformed) {
  CHECK_THROWS(parse(""), "unexpected end");
  CHECK_THROWS(parse("   "), "unexpected end");
  CHECK_THROWS(parse("{"), "unexpected end");
  CHECK_THROWS(parse("[1,]"), "trailing comma");
  CHECK_THROWS(parse(R"({"a":1,})"), "trailing comma");
  CHECK_THROWS(parse("[1 2]"), "expected ',' or ']'");
  CHECK_THROWS(parse(R"({"a" 1})"), "expected ':'");
  CHECK_THROWS(parse("{a:1}"), "expected a string key");
  CHECK_THROWS(parse("01"), "leading zeros");
  CHECK_THROWS(parse("1."), "after '.'");
  CHECK_THROWS(parse(".5"), "unexpected character");
  CHECK_THROWS(parse("+1"), "unexpected character");
  CHECK_THROWS(parse("1e"), "exponent");
  CHECK_THROWS(parse("-"), "missing digits");
  CHECK_THROWS(parse("NaN"), "unexpected character");
  CHECK_THROWS(parse("Infinity"), "unexpected character");
  CHECK_THROWS(parse("tru"), "invalid literal");
  CHECK_THROWS(parse("nul"), "invalid literal");
  CHECK_THROWS(parse("1e400"), "out of range");
  CHECK_THROWS(parse("1 2"), "trailing characters");
  CHECK_THROWS(parse("{} x"), "trailing characters");
  CHECK_THROWS(parse("\xEF\xBB\xBF{}"), "unexpected character");  // UTF-8 BOM not allowed
  CHECK_THROWS(parse("{\"a\":1}\n\n  ]"), "line 3, column 3");      // error position reported
}

TEST(json_rejects_bad_strings) {
  CHECK_THROWS(parse(R"("abc)"), "unterminated string");
  CHECK_THROWS(parse("\"a\nb\""), "control character");
  CHECK_THROWS(parse("\"a\tb\""), "control character");
  CHECK_THROWS(parse(R"("\x")"), "invalid escape");
  CHECK_THROWS(parse("\"\\u12\""), "hex digit");
  CHECK_THROWS(parse("\"\\u12G4\""), "hex digit");
  CHECK_THROWS(parse("\"\\uD800\""), "unpaired high surrogate");
  CHECK_THROWS(parse("\"\\uDC00\""), "unpaired low surrogate");
  CHECK_THROWS(parse("\"\\uD800\\u0041\""), "not followed by a low surrogate");
  CHECK_THROWS(parse("\"\xC3\x28\""), "invalid UTF-8");       // bad continuation
  CHECK_THROWS(parse("\"\xC0\xAF\""), "invalid UTF-8");       // overlong '/'
  CHECK_THROWS(parse("\"\xED\xA0\x80\""), "invalid UTF-8");   // UTF-8-encoded surrogate
  CHECK_THROWS(parse("\"\xF4\x90\x80\x80\""), "invalid UTF-8");  // above U+10FFFF
  CHECK_THROWS(parse("\"\xE2\x96\""), "invalid UTF-8");       // truncated sequence
}

TEST(json_duplicate_keys) {
  CHECK_THROWS(parse(R"({"a": 1, "a": 2})"), "duplicate key \"a\"");
  CHECK_THROWS(parse(R"({"x": {"k": 1, "k": 1}})"), "duplicate key \"k\"");
  ef::json::ParseOptions lax;
  lax.allow_duplicate_keys = true;
  const auto v = parse(R"({"a": 1, "a": 2})", lax);
  CHECK_EQ(v.size(), std::size_t{1});
  CHECK_EQ(v.at("a").as_int64(), 2);  // last wins when explicitly allowed
}

TEST(json_depth_limit) {
  std::string deep(300, '[');
  deep += std::string(300, ']');
  CHECK_THROWS(parse(deep), "nesting depth");
  std::string ok(200, '[');
  ok += std::string(200, ']');
  CHECK(parse(ok).is_array());
}

TEST(json_escape_roundtrip) {
  const std::string raw = "q\"b\\n\n\t\x01 ▁é😀";
  const std::string doc = "\"" + ef::json::escape(raw) + "\"";
  CHECK_EQ(parse(doc).as_string(), raw);
}

TEST(json_fuzz_never_crashes) {
  const std::string seed =
      R"({"model.layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [2048, 2048],)"
      R"( "data_offsets": [0, 8388608]}, "__metadata__": {"format": "pt"}, )"
      "\"u\": \"\\u00e9\\uD83D\\uDE00\""
      R"(, "n": -1.5e-3})";
  CHECK(parse(seed).is_object());
  std::mt19937 rng(12345);
  int parsed = 0, rejected = 0;
  for (int iter = 0; iter < 20000; ++iter) {
    std::string s = seed;
    const int edits = 1 + static_cast<int>(rng() % 4);
    for (int e = 0; e < edits; ++e) {
      const std::size_t pos = rng() % s.size();
      switch (rng() % 3) {
        case 0: s[pos] = static_cast<char>(rng() % 256); break;
        case 1: s.erase(pos, 1 + rng() % 3); break;
        default: s.insert(pos, 1, "{}[]\",:\\0e-"[rng() % 11]); break;
      }
      if (s.empty()) s = "x";
    }
    try {
      (void)parse(s);
      ++parsed;
    } catch (const ef::Error&) {
      ++rejected;
    }
  }
  CHECK(rejected > 0);
  CHECK_EQ(parsed + rejected, 20000);
}
