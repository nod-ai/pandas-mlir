cmake_minimum_required(VERSION 3.18)
project(python_ast_importer)

pybind11_add_module(python_ast_importer
  import_ast.cpp
  bindings.cpp
)

