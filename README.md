# pandas-mlir

# Requirements

Requires Python 3.9+. Install python requirements using pip.

```shell
$ pip install -r requirements.txt
```

# Build

```shell
$ ./cmake_configure.sh

# Build everything
$ cmake --build build

# Run unit tests
$ cmake --build build --target check-pandas-mlir

# Convert Pandas Python -> Pandas MLIR
$ cd python/test; pytest -s
```
