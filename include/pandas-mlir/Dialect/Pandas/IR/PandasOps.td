#ifndef PANDAS_OPS
#define PANDAS_OPS

include "pandas-mlir/Dialect/Pandas/IR/PandasTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Pandas_Op<string mnemonic, list<Trait> traits = []>
    : Op<Pandas_Dialect, mnemonic, traits> {
}

// TODO: Add extents of slices and row-slices
def Pandas_SliceOp : Pandas_Op<"slice", [NoSideEffect]> {
    let summary = "Slice a pandas.dataframe to get a pandas.series";
    let description = [{}];
    let arguments = (ins Pandas_DataFrameType:$dataframe, StrAttr:$column);
    let results = (outs Pandas_SeriesType:$slice);
    let assemblyFormat = [{
        $dataframe `,` $column attr-dict `:` qualified(type($dataframe)) `->` qualified(type($slice))
    }];
}

#endif // PANDAS_OPS
