# Installing from the binary package

## Debian and Ubuntu

The HDF5-UDF package and the syscall_intercept dependency are both hosted
on a PPA repository. Please run the following commands to include that PPA
on the list of sources searched by `apt`:

```
$ echo "deb https://lucasvr.gobolinux.org/debian/repo/ /" >> /etc/apt/sources.list.d/internal.list
$ wget -q -O - https://lucasvr.gobolinux.org/debian/repo/KEY.gpg | apt-key add -
```

And then install the binary packages with:

```
$ apt update
$ apt install libsyscall-intercept0
$ apt install hdf5-udf
```

# Building from the source code

## Requirements

HDF5-UDF comes with three backends, each of which requiring different
pieces of software to allow the embedding of bytecode and their execution:

- `Lua`: requires the [LuaJIT](https://luajit.org/install.html) package
- `Python`: requires the [CFFI](https://pypi.org/project/cffi) module
- `C/C++`: requires the [GNU C++ compiler](https://gnu.org/software/gcc)

It is possible to compile the code so that only a restricted number of system
calls can be executed by the user-defined functions. We rely on two packages
to limit what the UDF process can do:

- The [libseccomp](https://github.com/seccomp/libseccomp) library
- The [syscall_intercept](https://github.com/pmem/syscall_intercept) library

Please follow your distribution instructions to install these packages.
Also, make sure that you install both regular and development packages.


## Building the code

Simply run `make` followed by `make install`, optionally providing an alternative
destination directory other than `/usr/local`:

```
$ make
$ make install DESTDIR=/installation/path
```

By default, `make` will attempt to build all backends and to compile the HDF5
filter with support for system call filtering. It is possible to disable
features by providing the following arguments to `make`:

- `OPT_SANDBOX=0`: disable support for system call filtering
- `OPT_PYTHON=0`: disable Python backend
- `OPT_LUA=0`: disable Lua/LuaJIT backend
- `OPT_CPP=0`: disable C/C++ backend
