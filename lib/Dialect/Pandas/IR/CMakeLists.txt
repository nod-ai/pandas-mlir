add_mlir_library(PandasMLIRPandasDialect
  PandasDialect.cpp
  PandasOps.cpp
  PandasTypes.cpp
  PandasAttributes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/pandas-mlir/Dialect/Pandas

  DEPENDS
  MLIRPandasOpsIncGen
  MLIRPandasTypesIncGen
  MLIRPandasAttributesIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
