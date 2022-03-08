#ifndef PANDAS_BASE
#define PANDAS_BASE

include "mlir/IR/OpBase.td"

def Pandas_Dialect : Dialect {
  let name = "pandas";
  let cppNamespace = "::mlir::pandas::Pandas";
  let description = [{
  }];
}


#endif // PANDAS_BASE
