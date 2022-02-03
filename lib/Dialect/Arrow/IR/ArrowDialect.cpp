#include "pandas-mlir/Dialect/Arrow/IR/ArrowDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::pandas;
using namespace mlir::pandas::Arrow;

void ArrowDialect::initialize() {}

#include "pandas-mlir/Dialect/Arrow/IR/ArrowDialect.cpp.inc"
