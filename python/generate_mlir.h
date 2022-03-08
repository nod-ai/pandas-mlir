#pragma once

#include "pandas-mlir/Dialect/Pandas/IR/PandasAttributes.h"
#include "pandas-mlir/Dialect/Pandas/IR/PandasTypes.h"
#include "pandas-mlir/Dialect/Pandas/IR/PandasOps.h"
#include "pandas-mlir/Dialect/Pandas/IR/PandasDialect.h"
#include "pandas-mlir/InitAll.h"
#include "generate_mlir.h"
#include "import_ast.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"

class MLIRGenerator {
public:
  MLIRGenerator(mlir::MLIRContext &ctx) : builder(&ctx) {
    ctx.loadDialect<mlir::pandas::Pandas::PandasDialect,
                    mlir::StandardOpsDialect>();
  }
  void init();
  mlir::LogicalResult createFuncOp(const std::string_view &name,
                    const llvm::SmallVectorImpl<mlir::Type> &argTypes,
                    const llvm::SmallVectorImpl<mlir::Type> &retTypes,
                    const llvm::SmallVectorImpl<llvm::StringRef> &argNames);
  mlir::LogicalResult createSliceOp(const std::string_view &column,
                                    const llvm::StringRef &dataframe,
                                    const std::vector<std::string_view> &resultNames);
  mlir::LogicalResult createReturnOp(const std::string_view &returnName);
  void dump() {
    moduleOp.dump();
  }
  mlir::MLIRContext *context() {
    return builder.getContext();
  }

private:

  mlir::Location loc(const python::Location &loc);
  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value);
  mlir::ModuleOp moduleOp;
  mlir::OpBuilder builder;
  llvm::StringMap<mlir::Value> symbolTable;
};
