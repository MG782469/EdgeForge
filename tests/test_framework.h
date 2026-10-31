// EdgeForge — tiny self-contained test framework (no downloads needed offline).
#pragma once

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "ef/error.h"

namespace ef_test {

struct Case {
  std::string name;
  void (*fn)();
};

inline std::vector<Case>& registry() {
  static std::vector<Case> r;
  return r;
}

struct Registrar {
  Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

inline int& failures() {
  static int f = 0;
  return f;
}

inline long& checks() {
  static long c = 0;
  return c;
}

inline void fail(const char* file, int line, const std::string& msg) {
  ++failures();
  std::printf("    FAIL %s:%d: %s\n", file, line, msg.c_str());
}

template <class A, class B>
std::string describe(const A& a, const B& b) {
  std::ostringstream os;
  os << +a << " vs " << +b;
  return os.str();
}

inline std::string describe(const std::string& a, const std::string& b) { return "\"" + a + "\" vs \"" + b + "\""; }

}  // namespace ef_test

#define TEST(name)                                                  \
  static void name();                                               \
  static const ::ef_test::Registrar name##_registrar(#name, name);  \
  static void name()

#define CHECK(cond)                                         \
  do {                                                      \
    ++::ef_test::checks();                                  \
    if (!(cond)) ::ef_test::fail(__FILE__, __LINE__, #cond); \
  } while (0)

#define CHECK_EQ(a, b)                                                                              \
  do {                                                                                              \
    ++::ef_test::checks();                                                                          \
    const auto ef_va = (a);                                                                         \
    const auto ef_vb = (b);                                                                         \
    if (!(ef_va == ef_vb)) {                                                                        \
      ::ef_test::fail(__FILE__, __LINE__,                                                           \
                      std::string(#a " == " #b "  (") + ::ef_test::describe(ef_va, ef_vb) + ")");   \
    }                                                                                               \
  } while (0)

// Expects ef::Error whose message contains `substr`.
#define CHECK_THROWS(expr, substr)                                                                   \
  do {                                                                                               \
    ++::ef_test::checks();                                                                           \
    bool ef_thrown = false;                                                                          \
    try {                                                                                            \
      (void)(expr);                                                                                  \
    } catch (const ::ef::Error& ef_e) {                                                              \
      ef_thrown = true;                                                                              \
      if (std::string(ef_e.what()).find(substr) == std::string::npos) {                             \
        ::ef_test::fail(__FILE__, __LINE__,                                                          \
                        std::string("message \"") + ef_e.what() + "\" does not contain \"" + (substr) + "\""); \
      }                                                                                              \
    }                                                                                                \
    if (!ef_thrown) ::ef_test::fail(__FILE__, __LINE__, "expected ef::Error from: " #expr);          \
  } while (0)
