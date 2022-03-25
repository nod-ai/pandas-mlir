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
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

class MLIRGenerator {
public:
  MLIRGenerator(mlir::MLIRContext &ctx) : builder(&ctx) {
    ctx.loadDialect<mlir::pandas::Pandas::PandasDialect,
                    mlir::func::FuncDialect>();
  }
  void init();
  mlir::LogicalResult createFuncOp(const std::string_view &name,
                    const llvm::SmallVectorImpl<mlir::Type> &argTypes,
                    const llvm::SmallVectorImpl<mlir::Type> &retTypes,
                    const llvm::SmallVectorImpl<llvm::StringRef> &argNames);
  llvm::Optional<mlir::Value> createSliceOp(mlir::Value dataframe, mlir::Value column,
                                            const llvm::StringRef &dataframeVar);
  llvm::Optional<mlir::Value> createBinOp(mlir::Value left, mlir::Value right,
                                          const llvm::StringRef &type);
  llvm::Optional<mlir::Value> createReturnOp(mlir::Value returnValue);
  llvm::Optional<mlir::Value> createStringConstantOp(const llvm::StringRef &string);
  llvm::Optional<mlir::Value> createIntConstantOp(const int &value);
  llvm::Optional<mlir::Value> createFloatConstantOp(const float &value);
  llvm::Optional<mlir::Value> lookup(const llvm::StringRef &var);
  mlir::LogicalResult runPasses();
  void addToSymbolTable(mlir::Value, const llvm::StringRef &var);
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
