# pandas-mlir

# Requirements

Requires Python 3.9+. Install python requirements using pip.

```shell
$ pip install -r requirements.txt
```

Also requires ccache.

On Linux, you can install using apt.
```shell
sudo apt install ccache
```

On MacOS, you can install using brew.
```shell
brew install ccache

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

# More information

See the wiki for more information on the project.
