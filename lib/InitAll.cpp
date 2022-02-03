#include "pandas-mlir/InitAll.h"

#include "mlir/IR/Dialect.h"
#include "pandas-mlir/Dialect/Arrow/IR/ArrowDialect.h"

namespace mlir::pandas {

void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::pandas::Arrow::ArrowDialect>();
}

void registerAllPasses() {}

}
