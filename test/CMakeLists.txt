llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(PANDAS_MLIR_TEST_DEPENDS
  FileCheck count not
  pandas-mlir-opt
)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND STANDALONE_TEST_DEPENDS StandalonePythonModules)
endif()

add_lit_testsuite(check-pandas-mlir
  "Running the pandas-mlir regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${PANDAS_MLIR_TEST_DEPENDS}
)
set_target_properties(check-pandas-mlir PROPERTIES FOLDER "Tests")

add_lit_testsuites(PANDAS_MLIR
    ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${PANDAS_MLIR_TEST_DEPENDS})
