add_subdirectory(PandasToLinalg)

add_mlir_library(PandasMLIRConversionPasses
  Passes.cpp

  DEPENDS
  PandasMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  PandasMLIRPandasToLinalg
)
