#pragma once

#include "mlir/IR/Dialect.h"

namespace mlir::pandas {
  void registerAllDialects(mlir::DialectRegistry &registry);
  void registerAllPasses();
}
