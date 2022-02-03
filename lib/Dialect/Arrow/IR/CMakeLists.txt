add_mlir_library(PandasMLIRArrowDialect
  ArrowDialect.cpp
  ArrowOps.cpp
  ArrowTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/pandas-mlir/Dialect/Arrow

  DEPENDS
  MLIRArrowOpsIncGen
  MLIRArrowTypesIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
