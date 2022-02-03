add_subdirectory(Dialect)

add_mlir_library(PandasMLIRInitAll
  InitAll.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  PandasMLIRArrowDialect
)
