import pandas as pd
import numpy as np
import time
import sys
sys.path.append('../python')
from pandas_exporter import export_pandas, annotate_pandas

@export_pandas
@annotate_pandas(
    arg0 = {"type": "DataFrame",
            "schema": {"a": "i32", "b": "i32"},
            "indices": ["key", 1, "hello"],
            "dims": [3, 2]},
    arg1 = {"type": "DataFrame",
            "schema": {"a": "i32", "b": "i32"},
            "indices": ["key", 1, "hello"],
            "dims": [3, 2]},
)
def compute(df0, df1):
    """
    This test function demonstrates how pandas-mlir can export functions
    when provided with complete information about the input.

    Args:
        df (pd.DataFrame): DataFrame with no nullable types and
        fixed size.

    Returns:
        A linear combination of the "a" column and "b" column.

    """
    # Perform an inner join on the 'a' column.
    # Should return the original dataframe because we
    return pd.merge(
        df0["a"],
        df1[["a", "b"]],
        left_on="a",
        right_on="a",
        # If true, adds a _merge column with details
        # about whether the row comes from left, right
        # or both columns we are joining on.
        indicator=False
    )

df = pd.DataFrame({"a": [1, 4, 9], "b": [0, 3, 2]})

startTime = time.time()
res = compute(df, df)
endTime = time.time()
print(res)
print("Elapsed time (ms): " + str((endTime - startTime) * 1000))
