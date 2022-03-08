cmake_minimum_required(VERSION 3.18)
project(pandas_mlir_converter)

pybind11_add_module(pandas_mlir_converter
  import_ast.cpp
  generate_mlir.cpp
  bindings.cpp
)

target_link_libraries(pandas_mlir_converter PRIVATE
  MLIRIR
  MLIRStandard
  PandasMLIRPandasDialect
)
