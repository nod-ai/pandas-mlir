#include <numeric>
#include <iostream>
#include "generate_mlir.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "pandas-mlir/Dialect/Pandas/IR/PandasOps.h"

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

Optional<Value> MLIRGenerator::createSliceOp(Value dataframe, Value column,
                                             const StringRef &dataframeVar) {
  auto dfType = dataframe.getType().cast<pandas::Pandas::DataFrameType>();
  auto schemaDictAttr = dfType.getSchema();
  Type retType;
  auto constantOp = column.getDefiningOp<pandas::Pandas::StringConstantOp>();
  StringRef columnVar = constantOp.value();
  std::string var = (dataframeVar.str() + "[" + columnVar.str() + "]");
  if (symbolTable.lookup(var))
    return symbolTable[var];

  for (auto &pair : schemaDictAttr.getSchema()) {
    if (pair.first == columnVar) {
      retType = pandas::Pandas::SeriesType::get(builder.getContext(), pair.second);
      break;
    }
  }
  Value result = builder.create<pandas::Pandas::SliceOp>(moduleOp.getLoc(), retType, dataframe, columnVar);
  if (failed(declare(var, result))) return llvm::None;
  return result;
}

Optional<Value> MLIRGenerator::createReturnOp(Value returnValue) {
  builder.create<func::ReturnOp>(moduleOp.getLoc(), returnValue);
  return llvm::None;
}

bool isPandasType(Type type) {
  return (type.isa<pandas::Pandas::DataFrameType>() ||
          type.isa<pandas::Pandas::SeriesType>());
}

Optional<Value> MLIRGenerator::createBinOp(Value left, Value right, const llvm::StringRef &type) {
  auto leftType = left.getType();
  auto rightType = right.getType();
  Type retType;
  if (isPandasType(leftType) || isPandasType(rightType)) {
    retType = isPandasType(leftType) ? leftType : rightType;
  }
  // TODO: Handle cases where both operands are not pandas dataframes or series
  Value result = builder.create<pandas::Pandas::BinaryOp>(moduleOp.getLoc(), retType, left, right, type);
  return result;
}

void MLIRGenerator::addToSymbolTable(mlir::Value value, const llvm::StringRef &var) {
  if (!symbolTable.count(var))
    symbolTable.insert({var, value});
}

llvm::Optional<mlir::Value> MLIRGenerator::createStringConstantOp(const llvm::StringRef &string) {
  if (symbolTable.lookup(string))
    return symbolTable[string];
  Value result = builder.create<pandas::Pandas::StringConstantOp>(moduleOp.getLoc(),
                        StringAttr::get(builder.getContext(), string));
  if (failed(declare(string, result))) return llvm::None;
  return result;
}

llvm::Optional<mlir::Value> MLIRGenerator::createIntConstantOp(const int &value) {
  auto string = std::to_string(value);
  if (symbolTable.lookup(string))
    return symbolTable[string];
  auto i32Type = IntegerType::get(builder.getContext(), 32);
  Value result = builder.create<arith::ConstantOp>(moduleOp.getLoc(),
                                                   IntegerAttr::get(i32Type, value));
  if (failed(declare(string, result))) return llvm::None;
  return result;
}

llvm::Optional<mlir::Value> MLIRGenerator::createFloatConstantOp(const float &value) {
  auto string = std::to_string(value);
  if (symbolTable.lookup(string))
    return symbolTable[string];
  auto f32Type = Float32Type::get(builder.getContext());
  Value result = builder.create<arith::ConstantOp>(moduleOp.getLoc(),
                                                   FloatAttr::get(f32Type, value));
  if (failed(declare(string, result))) return llvm::None;
  return result;
}

llvm::Optional<mlir::Value> MLIRGenerator::lookup(const llvm::StringRef &var) {
  if (symbolTable.lookup(var))
    return symbolTable[var];
  return llvm::None;
}

LogicalResult MLIRGenerator::runPasses() {
  PassManager pm(builder.getContext());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCSEPass());
  if (failed(pm.run(moduleOp))) {
    return moduleOp.emitError() << "failed to run passes";
  }
  return success();
}
