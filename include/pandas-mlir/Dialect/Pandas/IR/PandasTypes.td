#ifndef PANDAS_TYPES
#define PANDAS_TYPES

include "pandas-mlir/Dialect/Pandas/IR/PandasBase.td"
include "pandas-mlir/Dialect/Pandas/IR/PandasAttributes.td"

//===----------------------------------------------------------------------===//
// Type defs
//===----------------------------------------------------------------------===//

class Pandas_Type<string name, string typeMnemonic,
                  string baseCppClass = "::mlir::Type">
    : TypeDef<Pandas_Dialect, name, [], baseCppClass> {
  let mnemonic = typeMnemonic;
}

def Pandas_SeriesType: Pandas_Type<"Series", "series"> {
  let summary = "pandas.series";
  let description = [{
    One-dimensional ndarray with axis labels.
  }];
  let parameters = (ins "::mlir::Type":$type,
                        "ArrayAttr":$indices);
  let assemblyFormat = [{
    `<` $type `,` `index` `:` $indices `>`
  }];
}

def Pandas_DataFrameType: Pandas_Type<"DataFrame", "dataframe"> {
  let summary = "Two-dimensional potentially heterogeneous tabular data";
  let description = [{
      This type is based off the pandas' dataframe described here:
      https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html#pandas.DataFrame.
      We further impose a restriction that this type is immutable,
      thus imposing value semantics on any operator using this type.
  }];
  let parameters = (ins "SchemaDictAttr":$schema,
                        "ArrayAttr":$indices);
  let assemblyFormat = [{
    `<` $schema `,` `index` `:` $indices `>`
  }];
}

def Pandas_StringType: Pandas_Type<"String", "string"> {
  let summary = "pandas.string";
  let description = [{
    String type.
  }];
}

def AnyPandasType :
    AnyTypeOf<[Pandas_DataFrameType, Pandas_SeriesType, AnyTensor,
               AnyInteger, AnyFloat]>;

#endif // PANDAS_TYPES
