#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/MlirOptMain.h"
#include "pandas-mlir/InitAll.h"

using namespace mlir;

int main(int argc, char **argv) {
  registerAllPasses();
  mlir::pandas::registerAllPasses();
  DialectRegistry registry;
  registerAllDialects(registry);
  mlir::pandas::registerAllDialects(registry);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLIR modular optimizer driver\n", registry,
                        /*preloadDialectsInContext=*/false));
}
