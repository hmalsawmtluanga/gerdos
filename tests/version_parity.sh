#!/bin/sh
# Version parity: pyproject dynamic version must equal version.hpp.
# Fails loudly the moment they diverge.
set -e
SRC="$1"
# Three-way parity: the header, the setup.py resolution, and the
# independent CMake PROJECT_VERSION pin must all agree. setup.py reads
# the header, so header-vs-setup alone is tautological — CMake is the
# outside witness.
HEADER="$SRC/include/gerdos/version.hpp"
MAJOR=$(grep GERDOS_VERSION_MAJOR "$HEADER" | awk '{print $3}')
MINOR=$(grep GERDOS_VERSION_MINOR "$HEADER" | awk '{print $3}')
PATCH=$(grep GERDOS_VERSION_PATCH "$HEADER" | awk '{print $3}')
EXPECTED="$MAJOR.$MINOR.$PATCH"
GOT=$(cd "$SRC" && python3 setup.py --version 2>/dev/null)
CMAKE_PIN=$(grep -A 3 "^project(" "$SRC/CMakeLists.txt" | grep VERSION | awk '{print $2}')
if [ "$GOT" != "$EXPECTED" ]; then
  echo "FAIL: package version $GOT != header $EXPECTED"
  exit 1
fi
if [ "$CMAKE_PIN" != "$EXPECTED" ]; then
  echo "FAIL: CMake pin $CMAKE_PIN != header $EXPECTED"
  exit 1
fi
echo "version parity: $GOT (cmake pin $CMAKE_PIN)"
