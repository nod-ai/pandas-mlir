// RUN: pandas-mlir-opt <%s -convert-pandas-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @compute(
// CHECK-SAME:       %[[ARG0:.+]]: tensor<2x3xi32>) -> tensor<3xi32> {
// CHECK:            %[[V0:.+]] = tensor.extract_slice %[[ARG0]][0, 0] [1, 3] [1, 1] : tensor<2x3xi32> to tensor<3xi32>
// CHECK:            return %[[V0]] : tensor<3xi32>
#schema0 = #pandas.schema.dict<"a" : tensor<3xi32>, "b" : tensor<3xi32>>
module {
    func @compute(%arg0 : !pandas.dataframe<#schema0, index : [0, 1, 2]>) -> !pandas.series<tensor<3xi32>, index : [0, 1, 2]> {
        %0 = pandas.slice %arg0, "a" : !pandas.dataframe<#schema0, index : [0, 1, 2]> -> !pandas.series<tensor<3xi32>, index : [0, 1, 2]>
        return %0 : !pandas.series<tensor<3xi32>, index : [0, 1, 2]>
    }
}
