import pandas as pd
import time

def compute(df):
    res = df[(df["a"] >= 2) & (df["b"].str.len() > 3)]
    return res

df = pd.DataFrame({"a": [1, 4, 9], "b": ["hello", "world", None]})

startTime = time.time()
compute(df)
endTime = time.time()
print("Elapsed time (ms): " + str((endTime - startTime) * 1000))
