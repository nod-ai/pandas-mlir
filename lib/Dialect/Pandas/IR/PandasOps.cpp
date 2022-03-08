#include "pandas-mlir/Dialect/Pandas/IR/PandasOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"

#define GET_OP_CLASSES
#include "pandas-mlir/Dialect/Pandas/IR/PandasOps.cpp.inc"
