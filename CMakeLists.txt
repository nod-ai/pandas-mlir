cmake_minimum_required(VERSION 3.18)

project(pandas-mlir LANGUAGES CXX C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

# Include pybind11
find_package(Python3 COMPONENTS Interpreter Development)
add_subdirectory(external/pybind11)

# In-tree build with LLVM_EXTERNAL_PROJECTS=pandas-mlir
set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include)
set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")

set(PANDAS_MLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(PANDAS_MLIR_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)
set(MLIR_TABLEGEN_EXE mlir-tblgen)

include(TableGen)
include(AddLLVM)
include(AddMLIR)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(python)
add_subdirectory(tools)
add_subdirectory(test)
