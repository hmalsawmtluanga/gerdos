#!/bin/sh
# Consumer smoke test: the installable package must serve a find_package
# consumer (headers + include path). Fails loudly on doubled trees or
# empty include variables. Runs under ctest with the build dir as $1.
set -e
SRC="$1"
STAGE="${TMPDIR:-/tmp}/gerdos-smoke-stage"
CONSUMER="${TMPDIR:-/tmp}/gerdos-smoke-consumer"
BUILD="${TMPDIR:-/tmp}/gerdos-smoke-build"
rm -rf "$STAGE" "$CONSUMER" "$BUILD"
cmake --install "$SRC" --prefix "$STAGE" > /dev/null
mkdir -p "$CONSUMER"
cat > "$CONSUMER/CMakeLists.txt" <<EOF2
cmake_minimum_required(VERSION 3.20)
project(gerdos_smoke)
find_package(gerdos REQUIRED PATHS "$STAGE/lib/cmake/gerdos" NO_DEFAULT_PATH)
add_executable(smoke smoke.cpp)
target_include_directories(smoke PRIVATE \${gerdos_INCLUDE_DIRS})
target_compile_features(smoke PRIVATE cxx_std_20)
EOF2
cat > "$CONSUMER/smoke.cpp" <<EOF2
#include "gerdos/core/operation.hpp"
int main() {
    gerdos::WorkDescription work{4, 1, 0.0f, 1.0f, 0.0f};
    return work.valid() ? 0 : 1;
}
EOF2
cmake -S "$CONSUMER" -B "$BUILD" > /dev/null
cmake --build "$BUILD" -j > /dev/null
"$BUILD/smoke"
rm -rf "$STAGE" "$CONSUMER" "$BUILD"
