add_llvm_executable(pandas-mlir-opt pandas-mlir-opt.cpp)

install(TARGETS pandas-mlir-opt
EXPORT PandasMLIRTargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT pandas-mlir-opt)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(pandas-mlir-opt PRIVATE
  MLIROptLib
  PandasMLIRInitAll
  PandasMLIRArrowDialect
  ${dialect_libs}
  ${conversion_libs}
)
