#include "pandas-mlir/Dialect/Pandas/IR/PandasAttributes.h"
#include "pandas-mlir/Dialect/Pandas/IR/PandasTypes.h"
#include "pandas-mlir/Dialect/Pandas/IR/PandasOps.h"
#include "pandas-mlir/Dialect/Pandas/IR/PandasDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::pandas;
using namespace mlir::pandas::Pandas;

#include "pandas-mlir/Dialect/Pandas/IR/PandasDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Tablegen Attribute Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "pandas-mlir/Dialect/Pandas/IR/PandasAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// Tablegen Type Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "pandas-mlir/Dialect/Pandas/IR/PandasTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//

void PandasDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "pandas-mlir/Dialect/Pandas/IR/PandasOps.cpp.inc"
  >();
  addTypes<
    #define GET_TYPEDEF_LIST
    #include "pandas-mlir/Dialect/Pandas/IR/PandasTypes.cpp.inc"
  >();
  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "pandas-mlir/Dialect/Pandas/IR/PandasAttributes.cpp.inc"
  >();
}
