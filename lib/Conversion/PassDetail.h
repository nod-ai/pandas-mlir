#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pandas {

#define GEN_PASS_CLASSES
#include "pandas-mlir/Conversion/Passes.h.inc"

} // namespace pandas
} // namespace mlir
