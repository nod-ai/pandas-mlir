add_mlir_conversion_library(PandasMLIRPandasToLinalg
  PandasToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/pandas-mlir/Conversion/PandasToLinalg

  DEPENDS
  PandasMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLinalg
  MLIRMath
  PandasMLIRPandasDialect
)

