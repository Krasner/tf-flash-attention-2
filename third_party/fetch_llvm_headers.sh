#!/bin/bash
# Fetches the LLVM headers that TensorFlow's XLA headers include (llvm/ADT/...).
# The TF pip wheel doesn't ship them. Only headers are needed (templates and
# declarations); nothing from LLVM is linked.
#
# The commit must match the one TF was built with, see
# tensorflow/third_party/llvm/workspace.bzl at the TF release tag.
# Defaults are for TF 2.19.0. Override with LLVM_COMMIT / LLVM_SHA256.
set -e

LLVM_COMMIT="${LLVM_COMMIT:-f8287f6c373fcf993643dd6f0e30dde304c1be73}"
LLVM_SHA256="${LLVM_SHA256:-add2841174abc79c45aa309bdf0cf631aa8f97e7a4df57dcfca57c60df27527f}"

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/llvm-headers"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading llvm-project@$LLVM_COMMIT"
curl -fL -o "$TMP/llvm.tar.gz" "https://github.com/llvm/llvm-project/archive/$LLVM_COMMIT.tar.gz"
echo "$LLVM_SHA256  $TMP/llvm.tar.gz" | sha256sum -c

SRC="llvm-project-$LLVM_COMMIT"
tar xzf "$TMP/llvm.tar.gz" -C "$TMP" "$SRC/llvm/include" "$SRC/cmake/Modules/LLVMVersion.cmake"
version() { sed -n "s/.*set(LLVM_VERSION_$1 \([0-9a-z]*\)).*/\1/p" "$TMP/$SRC/cmake/Modules/LLVMVersion.cmake"; }
MAJOR=$(version MAJOR); MINOR=$(version MINOR); PATCH=$(version PATCH)

rm -rf "$OUT"
mkdir -p "$OUT"
cp -r "$TMP/$SRC/llvm/include" "$OUT/include"
CFG="$OUT/include/llvm/Config"

# Generate the CMake-configured headers. Everything off except what a normal
# Linux release build has on. ABI-check enforcing is disabled since we don't
# link libLLVMSupport (it would leave llvm::DisableABIBreakingChecks undefined).
sed -e 's/#cmakedefine01 LLVM_ENABLE_ABI_BREAKING_CHECKS/#define LLVM_ENABLE_ABI_BREAKING_CHECKS 0\n#define LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING 1/' \
    -e 's/#cmakedefine01 LLVM_ENABLE_REVERSE_ITERATION/#define LLVM_ENABLE_REVERSE_ITERATION 0/' \
    "$CFG/abi-breaking.h.cmake" > "$CFG/abi-breaking.h"

sed -e 's/#cmakedefine01 \(LLVM_ENABLE_THREADS\|LLVM_HAS_ATOMICS\|LLVM_UNREACHABLE_OPTIMIZE\)$/#define \1 1/' \
    -e 's/#cmakedefine01 \([A-Z0-9_]*\)/#define \1 0/' \
    -e 's/#cmakedefine LLVM_ON_UNIX .*/#define LLVM_ON_UNIX 1/' \
    -e 's|#cmakedefine \([A-Z0-9_]*\).*|/* #undef \1 */|' \
    -e 's/\${LLVM_DEFAULT_TARGET_TRIPLE}//' \
    -e "s/\${LLVM_VERSION_MAJOR}/$MAJOR/" -e "s/\${LLVM_VERSION_MINOR}/$MINOR/" \
    -e "s/\${LLVM_VERSION_PATCH}/$PATCH/" -e "s/\${PACKAGE_VERSION}/$MAJOR.$MINOR.${PATCH}git/" \
    "$CFG/llvm-config.h.cmake" > "$CFG/llvm-config.h"

echo "LLVM $MAJOR.$MINOR.$PATCH headers in $OUT/include"
