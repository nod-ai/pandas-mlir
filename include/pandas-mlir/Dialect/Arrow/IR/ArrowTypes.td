#ifndef ARROW_TYPES
#define ARROW_TYPES

include "pandas-mlir/Dialect/Arrow/IR/ArrowBase.td"

class Arrow_Type<string name, string typeMnemonic,
                 string baseCppClass = "::mlir::Type">
    : TypeDef<Arrow_Dialect, name, [], baseCppClass> {
  let mnemonic = typeMnemonic;
}


#endif // ARROW_TYPES
