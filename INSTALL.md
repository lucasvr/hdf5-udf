# Installing from the binary package

## Debian and Ubuntu

The HDF5-UDF package and the syscall_intercept dependency are both hosted
on a PPA repository. Please run the following commands to include that PPA
on the list of sources searched by `apt`:

```
$ apt install -y wget gnupg
$ echo "deb https://lucasvr.gobolinux.org/debian/repo/ /" >> /etc/apt/sources.list.d/internal.list
$ wget -q -O - https://lucasvr.gobolinux.org/debian/repo/KEY.gpg | apt-key add -
```

And then install the binary packages (along with LuaJIT, so that you can
write and read UDFs written in the Lua language) with:

```
$ apt update
$ apt install -y hdf5-udf libsyscall-intercept0 luajit
```

Last, but not least, make sure that the `cffi` Python package is installed so
that Python UDFs can run as expected:

```
$ pip3 install cffi
```

## Fedora and RHEL

We host the HDF5-UDF and the syscall_intercept dependency on a YUM
repository. Please run the following commands to let your system
search for packages on that location:

```
$ wget -q -O /etc/pki/rpm-gpg/RPM-GPG-KEY-hdf5-udf https://lucasvr.gobolinux.org/fedora/repo/KEY.gpg
$ cat << EOF > /etc/yum.repos.d/hdf5-udf.repo
[HDF5-UDF]
name=HDF5-UDF RPM Server
baseurl=https://lucasvr.gobolinux.org/fedora/repo
enabled=1
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-hdf5-udf
EOF
```

And now you are ready to install the packages with YUM:

```
$ yum install -y hdf5-udf libsyscall_intercept
```

Last, please make sure that the `cffi` Python package is installed with:

```
$ pip3 install cffi
```

# Building from the source code

## Requirements

HDF5-UDF comes with three backends, each of which requiring different
pieces of software to allow the embedding of bytecode and their execution:

- `Lua`: requires the [LuaJIT](https://luajit.org/install.html) package
- `Python`: requires the [CFFI](https://pypi.org/project/cffi) module
- `C/C++`: requires the [GNU C++ compiler](https://gnu.org/software/gcc) or,
   alternatively, the [Clang compiler](https://clang.llvm.org).

It is possible to compile the code so that only a restricted number of system
calls can be executed by the user-defined functions. We rely on three packages
to limit what the UDF process can do:

- The [libseccomp](https://github.com/seccomp/libseccomp) library
- The [syscall_intercept](https://github.com/pmem/syscall_intercept) library
- The [libsodium](https://libsodium.gitbook.io) package, used for UDF signing
  and mapping of foreign public keys to `seccomp` rules

Please follow your distribution instructions to install these packages.
Also, make sure that you install both regular and development packages.


## Building the code

Simply run `make` followed by `make install`, optionally providing an alternative
destination directory other than `/usr/local`:

```
$ make
$ make install DESTDIR=/installation/path
```

In order to use an alternative compiler, please set the `CXX` variable to the
path to that executable (e.g., `CXX=clang++`).

By default, `make` will attempt to build all backends and to compile the HDF5
filter with support for system call filtering. It is possible to disable
features by providing the following arguments to `make`:

- `OPT_SANDBOX=0`: disable support for system call filtering
- `OPT_PYTHON=0`: disable Python backend
- `OPT_LUA=0`: disable Lua/LuaJIT backend
- `OPT_CPP=0`: disable C/C++ backend
