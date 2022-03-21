#ifndef PANDASMLIR_CONVERSION_PASSES
#define PANDASMLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertPandasToLinalg : Pass<"convert-pandas-to-linalg", "FuncOp"> {
  let summary = "Convert Pandas ops to Linalg ops";
  let constructor = "mlir::pandas::createConvertPandasToLinalgPass()";
}

#endif // PANDASMLIR_CONVERSION_PASSES
