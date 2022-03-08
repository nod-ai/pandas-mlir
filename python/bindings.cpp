#include "pybind11/pybind11.h"
#include "import_ast.h"
#include "generate_mlir.h"

PYBIND11_MODULE(pandas_mlir_converter, m) {
  m.doc() = "pandas-mlir Pandas MLIR converter";
  m.def("convert_to_mlir", &convert_to_mlir, "A function that converts specific functions in a python AST to MLIR");
}
