#include "import_ast.h"
#include "generate_mlir.h"
#include "pandas-mlir/Dialect/Pandas/IR/PandasAttributes.h"
#include "pandas-mlir/Dialect/Pandas/IR/PandasTypes.h"
#include "pandas-mlir/Dialect/Pandas/IR/PandasOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include <iostream>

static constexpr std::string_view PANDAS_EXPORT_ATTR = "export_pandas";
static constexpr std::string_view PANDAS_ANNOTATE_ATTR = "annotate_pandas";

using namespace mlir;
using namespace llvm;
using locMap = std::unordered_map<python::Location, py::list, python::LocHasher>;

struct Schema {
  std::vector<std::string> headers;
  std::vector<std::string> types;
};

std::string_view getName(py::handle obj) {
  return obj.attr("__class__").attr("__name__").cast<std::string_view>();
}

bool isASTType(py::handle node) {
  std::string_view moduleName{""};
  try {
    moduleName = node.attr("__class__").attr("__module__").cast<std::string_view>();
  } catch (const std::exception &e) {}
  return moduleName == "ast";
}

LogicalResult generate_mlir(py::handle node, MLIRGenerator &gen, locMap &symbolLocTable) {
  if (getName(node) == "Assign") {
    symbolLocTable[python::Location(node.attr("value"))] = node.attr("targets");
  }
  if (getName(node) == "Subscript") {
    auto column = node.attr("slice").attr("value").cast<std::string_view>();
    auto dataframe = node.attr("value").attr("id").cast<std::string_view>();
    python::Location loc(node);
    if (!symbolLocTable.count(loc)) {
      return failure();
    }
    py::list results = symbolLocTable[loc];
    std::vector<std::string_view> resultNames;
    for (size_t i = 0; i < results.size(); i++) {
      resultNames.push_back(results[i].attr("id").cast<std::string_view>());
    }
    if (failed(gen.createSliceOp(column, dataframe, resultNames)))
      return failure();
  }
  if (getName(node) == "Return") {
    py::handle returnValue = node.attr("value");
    if (getName(returnValue) == "Name") {
      auto returnName = returnValue.attr("id").cast<std::string_view>();
      if (failed(gen.createReturnOp(returnName))) {
        return failure();
      }
    }
  }
  return success();
}

void visit(py::handle obj, MLIRGenerator &gen, locMap &symbolLocTable) {
  py::tuple fields = obj.attr("_fields");
  for (py::handle field : fields) {
    std::string f = field.cast<std::string>();
    py::handle node = obj.attr(f.c_str());
    if (py::isinstance<py::list>(node)) {
      py::list nodeList = node.cast<py::list>();
      for (py::handle subnode : node) {
        if (isASTType(subnode)) {
          if (failed(generate_mlir(subnode, gen, symbolLocTable)))
            return;
          visit(subnode, gen, symbolLocTable);
        }
      }
    } else if (isASTType(node)) {
      if (failed(generate_mlir(node, gen, symbolLocTable)))
        return;
      visit(node, gen, symbolLocTable);
    }
  }
}

void parseSchema(py::handle schemaDict, Schema &schema) {
  py::list keys = schemaDict.attr("keys");
  py::list values = schemaDict.attr("values");
  for (size_t i = 0; i < keys.size(); i++) {
    schema.headers.push_back(keys[i].attr("value").cast<std::string>());
    schema.types.push_back(values[i].attr("value").cast<std::string>());
  }
}

void parseDims(py::list dimsList, SmallVectorImpl<int64_t> &dims) {
  for (size_t i = 0; i < dimsList.size(); i++) {
    dims.push_back(dimsList[i].attr("value").cast<int64_t>());
  }
}

Type getElementType(const std::string &type, MLIRGenerator &gen) {
  if (type == "i32") {
    return IntegerType::get(gen.context(), 32);
  }
  return Type();
}

Type constructDataFrame(const Schema &schema, const SmallVectorImpl<int64_t> &dims, MLIRGenerator &gen) {
  SmallVector<std::pair<std::string, Type>> schemaVec;
  for (size_t i = 0; i < schema.headers.size(); i++) {
    Type elType = getElementType(schema.types[i], gen);
    // Check if this is a ranked tensor type
    // TODO: Add error checking
    if (!dims.empty()) {
      auto columnTensorType = RankedTensorType::get(dims[0], elType);
      schemaVec.push_back(std::make_pair(schema.headers[i], columnTensorType));
    }
  }
  auto schemaDictType = mlir::pandas::Pandas::SchemaDictAttr::get(gen.context(), schemaVec);
  return mlir::pandas::Pandas::DataFrameType::get(gen.context(), schemaDictType);
}

Type constructSeries(const Schema &schema, const SmallVectorImpl<int64_t> &dims,
                     MLIRGenerator &gen) {
  Type type;
  // TODO: Add error checking
  if (!dims.empty()) {
    Type elType = getElementType(schema.types[0], gen);
    type = RankedTensorType::get(dims, elType);
  }
  return mlir::pandas::Pandas::SeriesType::get(gen.context(), type);
}

void constructTypes(const std::string_view &type, const Schema &schema, const SmallVectorImpl<int64_t> &dims,
                    SmallVectorImpl<Type> &mlirTypes, MLIRGenerator &gen) {
  Type mlirType;
  if (type == "DataFrame") {
    mlirType = constructDataFrame(schema, dims, gen);
  } else if (type == "Series") {
    mlirType = constructSeries(schema, dims, gen);
  }
  mlirTypes.push_back(mlirType);
}

void constructFuncTypes(py::handle decorator, SmallVectorImpl<Type> &argTypes,
                        SmallVectorImpl<Type> &retTypes,
                        MLIRGenerator &gen) {
  py::list annotations = decorator.attr("keywords");
  for (size_t i = 0; i < annotations.size(); i++) {
    std::string_view type;
    SmallVector<int64_t> dims;
    Schema schema;
    std::string_view arg = annotations[i].attr("arg").cast<std::string_view>();
    py::list keys = annotations[i].attr("value").attr("keys");
    py::list values = annotations[i].attr("value").attr("values");
    for (size_t j = 0; j < keys.size(); j++) {
      std::string_view v = keys[j].attr("value").cast<std::string_view>();
      if (v == "type") {
        type = values[j].attr("value").cast<std::string_view>();
      } else if (v == "schema") {
        parseSchema(values[j], schema);
      } else if (v == "dims") {
        parseDims(values[j].attr("elts"), dims);
      }
    }
    // Extract "arg" or "ret" annotation
    arg = arg.substr(0, 3);
    if (arg == "arg") {
      constructTypes(type, schema, dims, argTypes, gen);
    } else if (arg == "ret") {
      constructTypes(type, schema, dims, retTypes, gen);
    }
  }
}

void convert_to_mlir(py::object ast) {
  // Create initial MLIR function
  MLIRContext ctx;
  MLIRGenerator generator(ctx);
  generator.init();
  locMap symbolLocTable;

  py::list modules = ast.attr("body");
  // TODO: Assumes only single exported function
  py::handle exportedFunc;
  SmallVector<Type> argTypes, retTypes;
  SmallVector<StringRef> argNames;
  for (size_t i = 0; i < modules.size(); i++) {
    if (getName(modules[i]) == "FunctionDef") {
      // Check if function has been marked with the export_pandas decorator
      py::list decorators = modules[i].attr("decorator_list");
      bool exportDecorator{false};
      for (size_t j = 0; j < decorators.size(); j++) {
        auto nodeType = getName(decorators[j]);
        if ((nodeType == "Name") &&
            (decorators[j].attr("id").cast<std::string_view>() == PANDAS_EXPORT_ATTR)) {
          exportDecorator = true;
        }
        if ((nodeType == "Call") &&
            (decorators[j].attr("func").attr("id").cast<std::string_view>() == PANDAS_ANNOTATE_ATTR)) {
          constructFuncTypes(decorators[j], argTypes, retTypes, generator);
        }
      }
      // Get arg names
      py::list args = modules[i].attr("args").attr("args");
      for (size_t j = 0; j < args.size(); j++) {
        auto argName = args[j].attr("arg").cast<std::string_view>();
        argNames.push_back(argName);
      }
      if (exportDecorator) {
        exportedFunc = modules[i];
        break;
      }
    }
  }
  if (exportedFunc.is(py::none()))
    std::cout << "Did not find any functions to export! Exiting ..." << std::endl;

  // TODO: Depth first traversal of exported function
  auto funcName = exportedFunc.attr("name").cast<std::string_view>();
  if (failed(generator.createFuncOp(funcName, argTypes, retTypes, argNames))) return;
  visit(exportedFunc, generator, symbolLocTable);
  // TODO: MLIR conversion of AST
  generator.dump();
}
