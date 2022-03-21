#include "pandas-mlir/Conversion/PandasToLinalg/PandasToLinalg.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "pandas-mlir/Dialect/Pandas/IR/PandasDialect.h"
#include "pandas-mlir/Dialect/Pandas/IR/PandasOps.h"

using namespace mlir;
using namespace mlir::pandas;
using namespace mlir::pandas::Pandas;

#define DEBUG_TYPE "convert-pandas-to-linalg"

namespace {

class PandasTypeConverter : public TypeConverter {
public:
  PandasTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(convertPandasTypes);
  }
  // If each column is a ranked tensor with the same number of rows and
  // the same element type, then we can lift to a higher-dimensional tensor
  static Optional<Type> convertPandasTypes(Type type) {
    if (type.isa<DataFrameType>()) {
      auto dfType = type.cast<DataFrameType>();
      auto schemaDictAttr = dfType.getSchema();
      auto schema = schemaDictAttr.getSchema();
      bool dense = true;
      llvm::SmallVector<int64_t> shape;
      RankedTensorType t0, t1;
      for (size_t i = 0; i < schema.size(); i++) {
        if (!schema[i].second.isa<RankedTensorType>()) {
          dense = false;
          break;
        }
        if (i == 0) continue;
        t0 = schema[i-1].second.cast<RankedTensorType>();
        t1 = schema[i].second.cast<RankedTensorType>();
        if ((t0.getShape() != t1.getShape()) ||
            (t0.getElementType() != t1.getElementType())) {
          dense = false;
          break;
        }
      }
      if (dense) {
        llvm::SmallVector<int64_t> newShape;
        newShape.push_back(schema.size());
        for (auto val : t0.getShape()) newShape.push_back(val);
        return RankedTensorType::get(newShape, t0.getElementType());
      }
      // TODO: Return masked tensor when this is not the case
    }
    if (type.isa<SeriesType>()) {
      // Series types are converted to their underlying tensor type
      // TODO: Handle different labels for rows
      return type.cast<SeriesType>().getType();
    }
    return llvm::None;
  }
};

} // namespace

namespace {

class SliceOpLowering : public OpConversionPattern<Pandas::SliceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(Pandas::SliceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    ValueRange operands = adaptor.getOperands();
    auto newType = operands[0].getType();
    if (!newType.isa<RankedTensorType>()) {
      return failure();
    }
    auto rankedType = newType.cast<RankedTensorType>();
    auto rank = rankedType.getRank();
    auto shape = llvm::to_vector<4>(rankedType.getShape());
    auto dfType = op.dataframe().getType().cast<DataFrameType>();
    auto schema = dfType.getSchema().getSchema();
    llvm::SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    for (size_t i = 0; i < schema.size(); i++) {
      if (schema[i].first == op.column()) {
        offsets[i] = rewriter.getI64IntegerAttr(i);
      }
    }
    llvm::SmallVector<OpFoldResult> sizes;
    for (int64_t i = 0; i < rank; i++) {
      if (i == 0) {
        sizes.push_back(rewriter.getI64IntegerAttr(1));
      } else {
        sizes.push_back(rewriter.getI64IntegerAttr(shape[i]));
      }
    }
    llvm::SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    auto rankReducedType = RankedTensorType::get(shape[1], rankedType.getElementType());
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(op, rankReducedType, operands[0], offsets, sizes, strides);
    return success();
  }
};

} // namespace

namespace {

class ReturnOpLowering : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

} // namespace

namespace {

class ConvertPandasToLinalg : public ConvertPandasToLinalgBase<ConvertPandasToLinalg> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    FuncOp func = getOperation();
    MLIRContext *ctx = func->getContext();
    PandasTypeConverter converter;
    RewritePatternSet patterns(ctx);
    populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns, converter);

    // Convert pandas.slice
    patterns.add<SliceOpLowering, ReturnOpLowering>(converter, ctx);

    ConversionTarget target(*ctx);
    target.addIllegalDialect<PandasDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addDynamicallyLegalOp<FuncOp>(
      [&] (FuncOp op) { return converter.isSignatureLegal(op.getType()); }
    );
    target.addDynamicallyLegalOp<func::ReturnOp>(
      [&] (func::ReturnOp op) { return converter.isLegal(op.getOperandTypes()); }
    );
    if (failed(applyPartialConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::pandas::createConvertPandasToLinalgPass() {
  return std::make_unique<ConvertPandasToLinalg>();
}
