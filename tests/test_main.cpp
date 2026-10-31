// EdgeForge — test runner. Usage: ef_tests [suite-prefix]   (e.g. ef_tests json)
#include <chrono>
#include <exception>

#include "test_framework.h"

int main(int argc, char** argv) {
  const std::string prefix = argc > 1 ? std::string(argv[1]) + "_" : "";
  int ran = 0, failed_tests = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (const auto& c : ef_test::registry()) {
    if (!prefix.empty() && c.name.rfind(prefix, 0) != 0) continue;
    ++ran;
    const int before = ef_test::failures();
    try {
      c.fn();
    } catch (const std::exception& e) {
      ef_test::fail(__FILE__, __LINE__, std::string("uncaught exception: ") + e.what());
    } catch (...) {
      ef_test::fail(__FILE__, __LINE__, "uncaught non-std exception");
    }
    const bool ok = ef_test::failures() == before;
    if (!ok) ++failed_tests;
    std::printf("[%s] %s\n", ok ? " OK " : "FAIL", c.name.c_str());
  }
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (ran == 0) {
    std::printf("no tests match '%s'\n", prefix.c_str());
    return 1;
  }
  std::printf("\n%d tests, %ld checks, %d failed tests, %d failed checks (%.2f s)\n", ran, ef_test::checks(),
              failed_tests, ef_test::failures(), s);
  return ef_test::failures() == 0 ? 0 : 1;
}
