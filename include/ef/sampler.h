// EdgeForge - Stage 4: next-token sampler.
// Order follows Hugging Face generate(): repetition penalty -> (temperature <= 0: argmax)
// -> temperature -> top-k -> top-p (nucleus) -> sample with a seeded RNG.
#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace ef {

struct SamplerConfig {
  float temperature = 0.0f;         // <= 0: greedy (deterministic)
  int top_k = 0;                    // 0: disabled
  float top_p = 1.0f;               // 1: disabled
  float repetition_penalty = 1.0f;  // 1: disabled; HF rule: logit>0 ? logit/p : logit*p
  std::uint64_t seed = 0;
};

class Sampler {
 public:
  explicit Sampler(const SamplerConfig& cfg) : cfg_(cfg), rng_(cfg.seed) {}

  // logits: [vocab] (copied, not modified). history: tokens already in the context
  // (prompt + generated) for the repetition penalty.
  std::int32_t sample(const float* logits, std::size_t vocab, const std::vector<std::int32_t>& history);
  const SamplerConfig& config() const noexcept { return cfg_; }

 private:
  SamplerConfig cfg_;
  std::mt19937_64 rng_;
  std::vector<float> work_;
  std::vector<std::int32_t> idx_;
};

}  // namespace ef
