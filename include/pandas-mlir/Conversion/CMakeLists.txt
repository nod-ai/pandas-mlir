set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(PandasMLIRConversionPassIncGen)

add_mlir_doc(Passes PandasMLIRConversionPasses ./ -gen-pass-doc)
