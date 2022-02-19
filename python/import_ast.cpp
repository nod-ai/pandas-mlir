#include "import_ast.h"
#include <iostream>

bool isASTType(py::handle node) {
  std::string moduleName{""};
  try {
    moduleName = node.attr("__class__").attr("__module__").cast<std::string>();
  } catch (const std::exception &e) {}
  return moduleName == "ast";
}

std::string name(py::handle node) {
  return node.attr("__class__").attr("__name__").cast<std::string>();
}

void visit(py::handle obj) {
  py::tuple fields = obj.attr("_fields");
  for (py::handle field : fields) {
    std::string f = field.cast<std::string>();
    py::handle node = obj.attr(f.c_str());
    if (py::isinstance<py::list>(node)) {
      py::list nodeList = node.cast<py::list>();
      for (py::handle subnode : node) {
        if (isASTType(subnode)) {
          visit(subnode);
        }
      }
    } else if (isASTType(node)) {
      visit(node);
    }
  }
}

void import_ast(py::object ast) {
  py::list modules = ast.attr("body");
  py::handle exportedFunc;
  // TODO: Use decorator to identify which function to export
  for (size_t i = 0; i < modules.size(); i++) {
    std::string moduleName = modules[i].attr("__class__").attr("__name__").cast<std::string>();
    if (moduleName == "FunctionDef") {
      exportedFunc = modules[i];
      break;
    }
  }
  if (exportedFunc.is(py::none()))
    std::cout << "Did not find any functions to export! Exiting ..." << std::endl;
  // Create MLIR function
  // TODO: Depth first traversal of exported function
  visit(exportedFunc);
  // TODO: MLIR conversion of AST
}
