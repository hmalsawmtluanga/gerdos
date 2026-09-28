#!/bin/sh
# Version parity: pyproject dynamic version must equal version.hpp.
# Fails loudly the moment they diverge.
set -e
SRC="$1"
HEADER="$SRC/include/gerdos/version.hpp"
MAJOR=$(grep GERDOS_VERSION_MAJOR "$HEADER" | awk '{print $3}')
MINOR=$(grep GERDOS_VERSION_MINOR "$HEADER" | awk '{print $3}')
PATCH=$(grep GERDOS_VERSION_PATCH "$HEADER" | awk '{print $3}')
EXPECTED="$MAJOR.$MINOR.$PATCH"
GOT=$(cd "$SRC" && python3 setup.py --version 2>/dev/null)
if [ "$GOT" != "$EXPECTED" ]; then
  echo "FAIL: package version $GOT != header $EXPECTED"
  exit 1
fi
echo "version parity: $GOT"
