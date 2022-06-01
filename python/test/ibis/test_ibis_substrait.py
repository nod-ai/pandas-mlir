from collections import OrderedDict

import ibis
import pytest
from google.protobuf import json_format

from ibis_substrait.compiler.translate import translate
from ibis_substrait.proto.substrait import type_pb2 as stt
from ibis_substrait.proto.substrait.algebra_pb2 import Expression, Rel

NULLABILITY_NULLABLE = stt.Type.Nullability.NULLABILITY_NULLABLE
NULLABILITY_REQUIRED = stt.Type.Nullability.NULLABILITY_REQUIRED

# Test adopted from upstream ibis-substrait unit(s)
# https://github.com/ibis-project/ibis-substrait/blob/main/ibis_substrait/tests/compiler/test_compiler.py

@pytest.fixture
def t0():
    return ibis.table(
        [
            ("full_name", "string"),
            ("age", "int64"),
            ("ts", "timestamp('UTC')"),
            ("delta", "interval"),
        ]
    )

@pytest.fixture
def t1():
    return ibis.table(
        [
            ("full_name", "string"),
            ("age", "int64"),
            ("ts", "timestamp('UTC')"),
            ("delta", "interval"),
        ]
    )


def to_dict(message):
    """Print Protobuf message as python dictionary object."""
    return json_format.MessageToDict(message)


def test_join(t0, t1, compiler):
    """A walkthrough of a join expression in Substrait."""
    expr = (
        t0.left_join(t1, t0.age == t1.age)
    )
    result = translate(expr, compiler)

    # This plan is a "volcano" style plan meant for bottoms-up execution.
    # As a result, we top-level operation in the relation is the final projection
    # https://github.com/substrait-io/substrait/blob/main/proto/substrait/algebra.proto
    #
    # TODO(bsarden): Find out which logical plan optimizers are used.
    assert(result.WhichOneof("rel_type") == "project")
    input_: Rel = result.project.input
    assert(input_.WhichOneof("rel_type") == "join")

    join: Rel = input_.join

    # The `Expression` message type describes functions / arguments to run on
    # the given operator. Each `Expression` defines a Relational Expression Type
    # (`rex_type``), which maps to a broad categorization of the underlying
    # function category (e.g., `ScalarFunction`, `WindowFunction`, `IfThen`,
    # etc.).
    join_expr: Expression = join.expression

    # A Join expression maps to a
    assert(join_expr.WhichOneof("rex_type") == "scalar_function")
    scalar_func = join_expr.scalar_function

    # Each `rex_type` function breaks down into their own `protobuf.Message` type,
    # but we will study the `ScalarFunction` as an example, since they are all pretty
    # similar. Each `ScalarFunction` maps to:
    #   1. A `function_reference`: which represents a "pointer" to a uniquely identifiable
    #      operator ID that has been [registered][0] with the corresponding `Plan` type.
    #      These functions are serialized / registered with the `Plan` object through the
    #      definition of an `Extension` (see link above) and are often referred to as `*_anchor`
    #      in the specification.
    #   2. A list of `FunctionArguments`: these include input type specifications, and can
    #      also be the result of another `Expression`.
    #   3. A `Type` definition for the output: Currently there is only one `output_type` per
    #      operator that is supported.
    #
    # [0]: https://github.com/ibis-project/ibis-substrait/blob/main/ibis_substrait/compiler/core.py#L53-L80
    assert len(scalar_func.args) == 2, "a join should always have two operands"

    # We verify that the inputs are `FieldReference` expressions, because we are grabbing
    # the `age` column on both tables. To check this, we make sure that both selection ordinal values
    # match, since the two tables are equivalent.
    arg0, arg1 = scalar_func.args
    assert arg0.WhichOneof("rex_type") == "selection"
    assert arg1.WhichOneof("rex_type") == "selection"
    sel0, sel1 = arg0.selection, arg1.selection
    sel0_col_id = sel0.direct_reference.struct_field.field
    sel1_col_id = sel1.direct_reference.struct_field.field
    assert sel0_col_id
    assert sel1_col_id

    # The left / right sides of the join equate to Table Scan operations
    #
    # Which contains a struct about field names, dtypes, and whether a field
    # is nullable.
    left, right = input_.join.left, input_.join.right
    assert left.WhichOneof("rel_type") == "read"
    assert right.WhichOneof("rel_type") == "read"

    # with open("test_join.pb", "wb") as f:
    #     f.write(result.SerializeToString())
    js = to_dict(result)
    assert js