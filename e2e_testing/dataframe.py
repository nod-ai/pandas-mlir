import pandas as pd
import numpy as np
import time
import sys
sys.path.append('../python')
from pandas_exporter import export_pandas, annotate_pandas

def test(df):
    return df["a"]

@export_pandas
@annotate_pandas(
    arg0 = {"type": "DataFrame",
            "schema": {"a": "i32", "b": "i32"},
            "indices": ["key", 1, "hello"],
            "dims": [3, 2]},
)
def compute(df):
    """
    This test function demonstrates how pandas-mlir can export functions
    when provided with complete information about the input.

    Args:
        df (pd.DataFrame): DataFrame with no nullable types and
        fixed size.

    Returns:
        A linear combination of the "a" column and "b" column.

    """
    x = df["a"] * 2 + df["a"]
    res = x + df["b"] * 3
    return res + x

df = pd.DataFrame({"a": [1, 4, 9], "b": [0, 3, 2]})

startTime = time.time()
res = compute(df)
endTime = time.time()
print("Elapsed time (ms): " + str((endTime - startTime) * 1000))
