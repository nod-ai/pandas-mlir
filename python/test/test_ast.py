import pytest
import sys
sys.path.append('../../build/lib')
import ast
import astunparse
import python_ast_importer

class TestCase:
    def test_convert(self):
        with open('../../e2e_testing/dataframe.py', 'r') as f:
            input = f.read()
        tree = ast.parse(input)
        python_ast_importer.import_ast(tree)
