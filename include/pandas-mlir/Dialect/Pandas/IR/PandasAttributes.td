#ifndef PANDAS_ATTRIBUTES
#define PANDAS_ATTRIBUTES

include "pandas-mlir/Dialect/Pandas/IR/PandasBase.td"

//===----------------------------------------------------------------------===//
// Attribute defs
//===----------------------------------------------------------------------===//

class Pandas_Attribute<string name, string attrMnemonic,
                 string baseCppClass = "::mlir::Attribute">
    : AttrDef<Pandas_Dialect, name, [], baseCppClass> {
  let mnemonic = attrMnemonic;
}

def Pandas_SchemaDictAttr : Pandas_Attribute<"SchemaDict", "schema.dict"> {
  let summary = "Dictionary representation of tabular schema";
  let description = [{
      This type is used to define the the headers and element types of
      each column in the dataframe when specified as a dictionary.
  }];
  let parameters = (ins ArrayRefParameter<"std::pair<std::string, Type>", "">:$schema);
  let printer = [{
    $_printer << "<" ;
    bool first = true;
    for (auto& pair : getImpl()->schema) {
        if (!first) $_printer << ",";
        $_printer << pair.first << ": " << pair.second;
        first = false;
    }
    $_printer << ">" ;
  }];
  let parser = [{
    SmallVector<std::pair<std::string, Type>> schema;
    if ($_parser.parseLess())
        return Attribute();
    do {
      std::string str;
      Type type;
      if ($_parser.parseString(&str) || $_parser.parseColon()
          || $_parser.parseType(type))
        return Attribute();
      schema.push_back(std::make_pair(str, type));
    } while (!$_parser.parseOptionalComma());
    if ($_parser.parseGreater())
        return Attribute();
    return get($_ctxt, schema);
  }];
}

#endif // PANDAS_ATTRIBUTES
