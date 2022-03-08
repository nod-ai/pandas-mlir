#include <numeric>
#include <iostream>
#include "generate_mlir.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;

void MLIRGenerator::init() {
  moduleOp = ModuleOp::create(builder.getUnknownLoc());
}

mlir::Location MLIRGenerator::loc(const python::Location &loc) {
  return FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.lineno, loc.col_offset);
}

mlir::LogicalResult MLIRGenerator::declare(llvm::StringRef var, mlir::Value value) {
  if (symbolTable.count(var))
    return failure();
  symbolTable.insert({var, value});
  return success();
}

// TODO: Handle case where result types are inferred not provided
mlir::LogicalResult MLIRGenerator::createFuncOp(const std::string_view &name,
                                 const llvm::SmallVectorImpl<mlir::Type> &argTypes,
                                 const llvm::SmallVectorImpl<mlir::Type> &retTypes,
                                 const llvm::SmallVectorImpl<llvm::StringRef> &argNames) {
  auto funcOp = FuncOp::create(moduleOp.getLoc(), name, builder.getFunctionType(argTypes, retTypes));
  auto &entryBlock = *funcOp.addEntryBlock();
  for (size_t i = 0; i < argTypes.size(); i++) {
    if (failed(declare(argNames[i], entryBlock.getArgument(i))))
      return failure();
  }
  builder.setInsertionPointToStart(&entryBlock);
  moduleOp.push_back(funcOp);
  return success();
}

mlir::LogicalResult MLIRGenerator::createSliceOp(const std::string_view &column,
                                                 const llvm::StringRef &dataframe,
                                                 const std::vector<std::string_view> &resultNames) {
  if (!symbolTable.count(dataframe)) {
    return failure();
  }
  auto df = symbolTable.lookup(dataframe);
  auto dfType = df.getType().cast<pandas::Pandas::DataFrameType>();
  auto schemaDictAttr = dfType.getSchema();
  Type retType;
  for (auto &pair : schemaDictAttr.getSchema()) {
    if (pair.first == column) {
      retType = pandas::Pandas::SeriesType::get(builder.getContext(), pair.second);
      break;
    }
  }
  auto sliceOp = builder.create<pandas::Pandas::SliceOp>(moduleOp.getLoc(), retType, df, column);
  // TODO: Handle multiple results
  if (failed(declare(resultNames[0], sliceOp.getResult()))) {
    return failure();
  }
  return success();
}

mlir::LogicalResult MLIRGenerator::createReturnOp(const std::string_view &returnName) {
  if (!symbolTable.count(returnName)) {
    return failure();
  }
  auto ret = symbolTable.lookup(returnName);
  builder.create<ReturnOp>(moduleOp.getLoc(), ret);
  return success();
}
