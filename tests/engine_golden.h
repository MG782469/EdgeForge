// EdgeForge - test helper: golden prompt data in the tools/dump_reference.py layout, produced by
// the verified BF16 engine (so tests can run the golden-based checks on tiny random models).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "ef/kernels.h"
#include "ef/model.h"

namespace ef_test {

// Writes <dir>/manifest.json + tensors: input_ids, embed_last, hidden_last, final_norm_last,
// logits_last, and with `full` also embed_all, hidden_all, logits_all. `greedy` > 0 adds the
// greedy continuation (manifest "greedy":{"ids":[...]}) and decode_logits for its first 2 steps.
inline void write_engine_golden(const std::filesystem::path& model_dir, const std::filesystem::path& dir,
                                const std::vector<std::int32_t>& ids, bool full, std::size_t greedy = 0) {
  using namespace ef;
  namespace fs = std::filesystem;
  const Model m = Model::load(model_dir);
  const ModelConfig& c = m.config();
  const auto D = static_cast<std::size_t>(c.hidden_size), L = static_cast<std::size_t>(c.num_hidden_layers);
  const auto V = static_cast<std::size_t>(c.vocab_size), S = ids.size();
  Transformer t(m, S + greedy + 1);
  ForwardTrace tr;
  tr.capture = true;
  std::vector<float> embed_all(S * D), hidden_all(L * S * D), final_last(D), logits_all(S * V);
  for (std::size_t s = 0; s < S; ++s) {
    kern::row_f32(embed_all.data() + s * D, m.embed_tokens(), static_cast<std::size_t>(ids[s]));
    const float* lg = t.forward(ids[s], s, true, &tr);
    for (std::size_t l = 0; l < L; ++l) {
      std::copy_n(tr.layer_out.data() + l * D, D, hidden_all.data() + (l * S + s) * D);
    }
    std::copy_n(lg, V, logits_all.data() + s * V);
    if (s + 1 == S) final_last = tr.final_norm;
  }
  // Greedy continuation through the KV cache, like dump_reference.py (argmax, stop at EOS).
  std::vector<std::int32_t> gen;
  std::vector<float> decode;
  std::vector<float> cur(logits_all.end() - static_cast<std::ptrdiff_t>(V), logits_all.end());
  for (std::size_t k = 0; k < greedy; ++k) {
    const auto tok = static_cast<std::int32_t>(kern::argmax(cur.data(), V));
    gen.push_back(tok);
    if (tok == static_cast<std::int32_t>(c.eos_token_id) || k + 1 == greedy) break;
    const float* lg = t.forward(tok, S + k, true);
    cur.assign(lg, lg + V);
    if (k < 2) decode.insert(decode.end(), lg, lg + V);
  }
  std::vector<float> embed_last(embed_all.end() - static_cast<std::ptrdiff_t>(D), embed_all.end()), hidden_last(L * D);
  for (std::size_t l = 0; l < L; ++l) std::copy_n(hidden_all.data() + (l * S + S - 1) * D, D, hidden_last.data() + l * D);
  fs::create_directories(dir);
  std::string files;
  auto put = [&](const std::string& key, const void* data, std::size_t bytes, const std::string& dtype,
                 const std::string& shape) {
    std::ofstream(dir / (key + ".bin"), std::ios::binary).write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    files += (files.empty() ? "" : ",") + std::string("\"") + key + "\":{\"file\":\"" + key + ".bin\",\"dtype\":\"" +
             dtype + "\",\"shape\":" + shape + "}";
  };
  const std::string d = std::to_string(D), l = std::to_string(L), s = std::to_string(S), v = std::to_string(V);
  put("input_ids", ids.data(), S * 4, "i32", "[" + s + "]");
  put("embed_last", embed_last.data(), D * 4, "f32", "[" + d + "]");
  put("hidden_last", hidden_last.data(), L * D * 4, "f32", "[" + l + "," + d + "]");
  put("final_norm_last", final_last.data(), D * 4, "f32", "[" + d + "]");
  put("logits_last", logits_all.data() + (S - 1) * V, V * 4, "f32", "[" + v + "]");
  if (full) {
    put("embed_all", embed_all.data(), S * D * 4, "f32", "[" + s + "," + d + "]");
    put("hidden_all", hidden_all.data(), L * S * D * 4, "f32", "[" + l + "," + s + "," + d + "]");
    put("logits_all", logits_all.data(), S * V * 4, "f32", "[" + s + "," + v + "]");
  }
  if (!decode.empty()) {
    put("decode_logits", decode.data(), decode.size() * 4, "f32", "[" + std::to_string(decode.size() / V) + "," + v + "]");
  }
  std::string g;
  for (auto id : gen) g += (g.empty() ? "" : ",") + std::to_string(id);
  std::ofstream(dir / "manifest.json") << "{\"format\":\"edgeforge-golden-v1\",\"name\":\"" << dir.filename().string()
                                       << "\",\"n_tokens\":" << S << ",\"greedy\":{\"ids\":[" << g << "]},\"files\":{"
                                       << files << "}}";
}

}  // namespace ef_test
