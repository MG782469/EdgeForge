cmake_minimum_required(VERSION 3.20)
project(EdgeForge VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(EF_NATIVE   "Optimise for this machine's CPU (-march=native / MSVC /arch:AVX2)" ON)
option(EF_WERROR   "Treat warnings as errors" OFF)
option(EF_SANITIZE "AddressSanitizer + UndefinedBehaviorSanitizer (GCC/Clang)" OFF)
set(EF_MODEL_DIR "" CACHE PATH "Optional TinyLlama folder; adds a real-checkpoint CTest")
set(EF_GOLDEN_DIR "" CACHE PATH "Optional golden folder from tools/dump_reference.py")

find_package(Threads REQUIRED)
include(CheckCXXCompilerFlag)

# ---------------------------------------------------------------- flags
set(EF_ARCH_FLAGS "")
if(EF_NATIVE)
  if(MSVC)
    set(EF_ARCH_FLAGS /arch:AVX2)
  else()
    check_cxx_compiler_flag(-march=native EF_HAS_MARCH_NATIVE)
    if(EF_HAS_MARCH_NATIVE)
      set(EF_ARCH_FLAGS -march=native)
    endif()
  endif()
endif()

if(MSVC)
  set(EF_WARN_FLAGS /W4 /permissive- /utf-8 /Zc:__cplusplus /D_CRT_SECURE_NO_WARNINGS)
  if(EF_WERROR)
    list(APPEND EF_WARN_FLAGS /WX)
  endif()
else()
  set(EF_WARN_FLAGS -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wcast-align
                    -Wformat=2 -Wdouble-promotion -Wimplicit-fallthrough)
  if(EF_WERROR)
    list(APPEND EF_WARN_FLAGS -Werror)
  endif()
endif()

# Reference kernels must be bit-reproducible across compilers: no implicit a*b+c -> FMA
# contraction (GCC < 13 contracts by default even in ISO mode; Clang contracts within
# expressions). MSVC's default /fp:precise already does not contract. Explicit FMA comes
# later, in the SIMD kernels, where it is tested against these.
set(EF_FP_FLAGS "")
if(NOT MSVC)
  set(EF_FP_FLAGS -ffp-contract=off)
endif()

set(EF_SAN_FLAGS "")
if(EF_SANITIZE AND NOT MSVC)
  set(EF_SAN_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" EF_BT_UPPER)
string(REPLACE ";" " " EF_ARCH_STR "${EF_ARCH_FLAGS}")
string(STRIP "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${EF_BT_UPPER}} ${EF_ARCH_STR} -ffp-contract=off" EF_FLAGS_STR)

function(ef_configure target)
  target_compile_options(${target} PRIVATE ${EF_WARN_FLAGS} ${EF_ARCH_FLAGS} ${EF_FP_FLAGS} ${EF_SAN_FLAGS})
  if(EF_SAN_FLAGS)
    target_link_options(${target} PRIVATE ${EF_SAN_FLAGS})
  endif()
endfunction()

# ---------------------------------------------------------------- core library
add_library(ef_core STATIC
  src/json.cpp
  src/mapped_file.cpp
  src/safetensors.cpp
  src/model_config.cpp
  src/membench.cpp
  src/sysinfo.cpp
  src/kernels.cpp
  src/golden.cpp
  src/verify_math.cpp
)
target_include_directories(ef_core PUBLIC include)
target_link_libraries(ef_core PUBLIC Threads::Threads)
if(WIN32)
  target_link_libraries(ef_core PUBLIC psapi)
endif()
target_compile_definitions(ef_core PRIVATE
  "EF_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\""
  "EF_BUILD_FLAGS=\"${EF_FLAGS_STR}\"")
ef_configure(ef_core)

# ---------------------------------------------------------------- CLI
add_executable(edgeforge src/main.cpp)
target_link_libraries(edgeforge PRIVATE ef_core)
ef_configure(edgeforge)

# ---------------------------------------------------------------- tests
enable_testing()
add_executable(ef_tests
  tests/test_main.cpp
  tests/test_json.cpp
  tests/test_fp16.cpp
  tests/test_safetensors.cpp
  tests/test_config.cpp
  tests/test_membench.cpp
  tests/test_kernels.cpp
  tests/test_golden.cpp
)
target_link_libraries(ef_tests PRIVATE ef_core)
target_include_directories(ef_tests PRIVATE tests)
ef_configure(ef_tests)

foreach(suite json fp16 safetensors config membench kernels golden)
  add_test(NAME ${suite} COMMAND ef_tests ${suite})
endforeach()

# Stage 2: kernels vs PyTorch golden outputs (weight-free checks need only tests/golden).
if(NOT EF_GOLDEN_DIR AND EXISTS "${CMAKE_SOURCE_DIR}/tests/golden/meta.json")
  set(EF_GOLDEN_DIR "${CMAKE_SOURCE_DIR}/tests/golden")
endif()
if(EF_GOLDEN_DIR)
  if(EF_MODEL_DIR)
    add_test(NAME golden_math COMMAND edgeforge verify-math --golden "${EF_GOLDEN_DIR}" --model-dir "${EF_MODEL_DIR}")
  else()
    add_test(NAME golden_math COMMAND edgeforge verify-math --golden "${EF_GOLDEN_DIR}")
  endif()
endif()

if(EF_MODEL_DIR)
  if(EF_GOLDEN_DIR)
    add_test(NAME real_checkpoint
             COMMAND edgeforge verify --model-dir "${EF_MODEL_DIR}" --golden "${EF_GOLDEN_DIR}")
  else()
    add_test(NAME real_checkpoint COMMAND edgeforge verify --model-dir "${EF_MODEL_DIR}")
  endif()
endif()
