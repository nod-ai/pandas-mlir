#include "pybind11/pybind11.h"
#include "import_ast.h"

namespace py = pybind11;

PYBIND11_MODULE(python_ast_importer, m) {
  m.doc() = "pandas-mlir python AST importer";
  m.def("import_ast", &import_ast, "A function that imports the python AST for a given module");
}
