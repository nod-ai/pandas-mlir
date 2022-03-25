import pytest
import sys
sys.path.append('../../build/lib')
import ast
import astunparse
import pandas_mlir_converter
from typing import Any

class MyVisitor(ast.NodeVisitor):
    def generic_visit(self, node):
        print(f'Nodetype: {type(node).__name__:{16}} {node}')
        if type(node) == ast.Assign:
            breakpoint()
        ast.NodeVisitor.generic_visit(self, node)

class TestCase:
    def test_convert(self):
        with open('../../e2e_testing/dataframe.py', 'r') as f:
            input = f.read()
        tree = ast.parse(input)
        #for node in ast.walk(tree):
        #    print(node, astunparse.unparse(node))
        #    if type(node) == ast.Assign:
        #        breakpoint()
        #v = MyVisitor()
        #v.visit(tree)
        #breakpoint()
        pandas_mlir_converter.convert_to_mlir(tree)
