// EdgeForge — common error type.
#pragma once

#include <stdexcept>
#include <string>

namespace ef {

// Every recoverable failure in EdgeForge throws ef::Error with a message that
// says what was wrong and, where possible, where (file, tensor, byte offset).
struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

}  // namespace ef
