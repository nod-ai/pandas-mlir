#!/usr/bin/env bash

set -e

  # -DCMAKE_C_COMPILER=clang \
  # -DCMAKE_CXX_COMPILER=clang++ \
# NOTE: We compile with gcc, but use `lld` because linkage is much
# faster on binaries with large / lots symbols (e.g. llvm/mlir) with
# `lld`.
#  -DCMAKE_CXX_FLAGS="-fuse-ld=lld" \
cmake -GNinja -Bbuild \
  -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_ENABLE_LIBCXX=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DLLVM_EXTERNAL_PROJECTS=pandas-mlir \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DLLVM_EXTERNAL_PANDAS_MLIR_SOURCE_DIR=`pwd` \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DLLVM_TARGETS_TO_BUILD=host \
  external/llvm-project/llvm