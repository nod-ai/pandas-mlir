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


    @pytest.mark.parametrize(
        "py_ast_path",
        [
            '../../e2e_testing/dataframe.py',
            pytest.param(
                '../../e2e_testing/join.py',
                marks=pytest.mark.xfail(reason='need error checking'),
            ),
        ]
    )
    def test_convert(self, py_ast_path):
        with open(py_ast_path, 'r') as f:
            input = f.read()
        tree = ast.parse(input)
        #for node in ast.walk(tree):
        #    print(node, astunparse.unparse(node))
        #    if type(node) == ast.Assign:
        #        breakpoint()
        #v = MyVisitor()
        #v.visit(tree)
        #breakpoint()
        # TODO: Error checking so that conversion
        pandas_mlir_converter.convert_to_mlir(tree)