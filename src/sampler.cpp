// EdgeForge - Stage 4 sampler implementation.
#include "ef/sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "ef/error.h"
#include "ef/kernels.h"

namespace ef {

std::int32_t Sampler::sample(const float* logits, std::size_t vocab, const std::vector<std::int32_t>& history) {
  if (vocab == 0) throw Error("sampler: empty vocabulary");
  work_.assign(logits, logits + vocab);
  if (cfg_.repetition_penalty != 1.0f) {
    std::vector<bool> seen(vocab, false);
    for (std::int32_t t : history) {
      if (t < 0 || static_cast<std::size_t>(t) >= vocab || seen[static_cast<std::size_t>(t)]) continue;
      seen[static_cast<std::size_t>(t)] = true;
      float& l = work_[static_cast<std::size_t>(t)];
      l = l > 0.0f ? l / cfg_.repetition_penalty : l * cfg_.repetition_penalty;
    }
  }
  if (cfg_.temperature <= 0.0f) return static_cast<std::int32_t>(kern::argmax(work_.data(), vocab));

  for (float& l : work_) l /= cfg_.temperature;
  idx_.resize(vocab);
  std::iota(idx_.begin(), idx_.end(), 0);
  std::size_t keep = vocab;
  auto by_logit = [&](std::int32_t a, std::int32_t b) {
    const float la = work_[static_cast<std::size_t>(a)], lb = work_[static_cast<std::size_t>(b)];
    return la != lb ? la > lb : a < b;
  };
  if (cfg_.top_k > 0 && static_cast<std::size_t>(cfg_.top_k) < vocab) {
    keep = static_cast<std::size_t>(cfg_.top_k);
    std::partial_sort(idx_.begin(), idx_.begin() + static_cast<std::ptrdiff_t>(keep), idx_.end(), by_logit);
  } else {
    std::sort(idx_.begin(), idx_.end(), by_logit);
  }
  // softmax over the kept candidates (sorted, highest first)
  const float mx = work_[static_cast<std::size_t>(idx_[0])];
  std::vector<double> p(keep);
  double sum = 0.0;
  for (std::size_t i = 0; i < keep; ++i) {
    p[i] = std::exp(static_cast<double>(work_[static_cast<std::size_t>(idx_[i])] - mx));
    sum += p[i];
  }
  for (double& v : p) v /= sum;
  if (cfg_.top_p < 1.0f) {
    double cum = 0.0;
    std::size_t n = 0;
    while (n < keep) {
      cum += p[n++];
      if (cum >= static_cast<double>(cfg_.top_p)) break;
    }
    keep = std::max<std::size_t>(n, 1);
    p.resize(keep);
    const double s2 = std::accumulate(p.begin(), p.end(), 0.0);
    for (double& v : p) v /= s2;
  }
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const double r = u(rng_);
  double cum = 0.0;
  for (std::size_t i = 0; i < keep; ++i) {
    cum += p[i];
    if (r < cum) return idx_[i];
  }
  return idx_[keep - 1];
}

}  // namespace ef
