# Requirements

HDF5-UDF comes with three backends, each of which requiring different
pieces of software to allow the embedding of bytecode and their execution:

- `Lua`: requires the [LuaJIT](https://luajit.org/install.html) package
- `Python`: requires the [CFFI](https://pypi.org/project/cffi) module
- `C/C++`: requires the [GNU C++ compiler](https://gnu.org/software/gcc)

Please follow your distribution instructions to install these packages.
Also, make sure that you install both regular and development packages.
