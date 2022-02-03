#ifndef ARROW_OPS
#define ARROW_OPS

include "pandas-mlir/Dialect/Arrow/IR/ArrowTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Arrow_Op<string mnemonic, list<Trait> traits = []>
    : Op<Arrow_Dialect, mnemonic, traits> {
}

#endif // ARROW_OPS
