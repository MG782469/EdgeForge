#!/usr/bin/env bash
# EdgeForge - build + test WITHOUT CMake (g++ or clang++ only).
# Usage: tools/build.sh [build-dir]        (default: ./build-sh)
#        CXX=clang++ tools/build.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/build-sh}"
CXX="${CXX:-g++}"
ARCH="${EF_ARCH:--march=native}"
mkdir -p "$OUT/obj"

FLAGS=(-std=c++20 -O3 -DNDEBUG "$ARCH" -ffp-contract=off -Wall -Wextra -Wpedantic -Wshadow -pthread -I"$ROOT/include")
DEFS=("-DEF_BUILD_TYPE=\"Release\"" "-DEF_BUILD_FLAGS=\"-O3 -DNDEBUG $ARCH -ffp-contract=off (tools/build.sh)\"")

echo "compiler: $("$CXX" --version | head -1)"
objs=()
for s in json mapped_file safetensors model_config membench sysinfo kernels golden verify_math model verify_forward tokenizer sampler chat verify_tokenizer quant qfile quantizer verify_quant qkernels threadpool eval_quant bench_decode; do
  echo "  CXX src/$s.cpp"
  "$CXX" "${FLAGS[@]}" "${DEFS[@]}" -c "$ROOT/src/$s.cpp" -o "$OUT/obj/$s.o"
  objs+=("$OUT/obj/$s.o")
done
echo "  LINK edgeforge"
"$CXX" "${FLAGS[@]}" "$ROOT/src/main.cpp" "${objs[@]}" -o "$OUT/edgeforge"
echo "  LINK ef_tests"
"$CXX" "${FLAGS[@]}" -I"$ROOT/tests" "$ROOT"/tests/test_*.cpp "${objs[@]}" -o "$OUT/ef_tests"
echo "built: $OUT/edgeforge  $OUT/ef_tests"

if [[ "${EF_SKIP_TESTS:-0}" != "1" ]]; then
  "$OUT/ef_tests"
fi
