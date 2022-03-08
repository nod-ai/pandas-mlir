#pragma once

#include <memory>
#include "pybind11/pybind11.h"
namespace py = pybind11;

namespace python {
struct Location {
  Location(py::handle node) {
    lineno = node.attr("lineno").cast<int>();
    col_offset = node.attr("col_offset").cast<int>();
    end_lineno = node.attr("end_lineno").cast<int>();
    end_col_offset = node.attr("end_col_offset").cast<int>();
  }
  bool operator==(const Location &loc) const {
    return (lineno == loc.lineno) && (col_offset == loc.col_offset)
           && (end_lineno == loc.end_lineno) && (end_col_offset == loc.end_col_offset);
  }
  int lineno;
  int col_offset;
  int end_lineno;
  int end_col_offset;
  std::shared_ptr<std::string> file;
};

struct LocHasher {
  std::size_t operator() (const Location &loc) const {
    return ((std::hash<int>()(loc.lineno)) ^
            (std::hash<int>()(loc.col_offset) << 1) ^
            (std::hash<int>()(loc.end_lineno) << 2) ^
            (std::hash<int>()(loc.end_col_offset) << 3));
  }
};

}

void convert_to_mlir(py::object ast);
