#include "pandas-mlir/Conversion/Passes.h"
#include "pandas-mlir/Conversion/PandasToLinalg/PandasToLinalg.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "pandas-mlir/Conversion/Passes.h.inc"
}

void mlir::pandas::registerConversionPasses() { ::registerPasses(); }
