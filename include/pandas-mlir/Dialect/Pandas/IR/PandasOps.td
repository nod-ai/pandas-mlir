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

def Pandas_StringConstantOp : Pandas_Op<"string.constant", [NoSideEffect]> {
    let summary = "Create a string constant";
    let description = [{}];
    let arguments = (ins StrAttr:$value);
    let results = (outs Pandas_StringType:$string);
    let assemblyFormat = [{
        attr-dict `:` qualified(type($string))
    }];
}

def Pandas_BinaryOp : Pandas_Op<"binaryop", [NoSideEffect]> {
    let summary = "Perform a binary operation between two values";
    let description = [{}];
    let arguments = (ins AnyPandasType:$left, AnyPandasType:$right,
                         StrAttr:$type);
    let results = (outs AnyPandasType:$res);
    let assemblyFormat = [{
        $left `,` $right attr-dict `:` `(` qualified(type($left)) `,` qualified(type($right)) `)` `->` qualified(type($res))
    }];
}

#endif // PANDAS_OPS
