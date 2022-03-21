#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::pandas { 
std::unique_ptr<OperationPass<FuncOp>> createConvertPandasToLinalgPass();
}
