import pytest
import sys
sys.path.append('../../build/lib')
import ast
import astunparse
import pandas_mlir_converter
from typing import Any

class TestCase:
    def test_convert(self):
        with open('../../e2e_testing/dataframe.py', 'r') as f:
            input = f.read()
        tree = ast.parse(input)
        pandas_mlir_converter.convert_to_mlir(tree)
