#ifndef ARROW_BASE
#define ARROW_BASE

include "mlir/IR/OpBase.td"

def Arrow_Dialect : Dialect {
  let name = "arrow";
  let cppNamespace = "::mlir::pandas::Arrow";
  let description = [{
  }];
}


#endif // ARROW_BASE
