# Installing from the binary package

## Debian and Ubuntu

The HDF5-UDF package is hosted on a PPA repository. Please run the following
commands to include that PPA on the list of sources searched by `apt`:

```
$ apt install -y wget gnupg
$ echo "deb https://lucasvr.gobolinux.org/debian/repo/ /" >> /etc/apt/sources.list.d/internal.list
$ wget -q -O - https://lucasvr.gobolinux.org/debian/repo/KEY.gpg | apt-key add -
```

And then install the binary package (along with LuaJIT, if you want to
write and read UDFs written in the Lua language) with:

```
$ apt update
$ apt install -y hdf5-udf luajit
```

Last, but not least, make sure that the `cffi` Python package is installed so
that Python UDFs can run as expected:

```
$ pip3 install cffi
```

## Fedora and RHEL

We host the HDF5-UDF package on a YUM repository. Please run the following
commands to let your system search for packages on that location:

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

And now you are ready to install the package with YUM:

```
$ yum install -y hdf5-udf
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
calls can be executed by the user-defined functions. We rely on two packages
to limit what the UDF process can do:

- The [libseccomp](https://github.com/seccomp/libseccomp) library
- The [libsodium](https://libsodium.gitbook.io) package, used for UDF signing
  and mapping of foreign public keys to `seccomp` rules

Please follow your distribution instructions to install these packages.
Also, make sure that you install both regular and development packages.


## Building the code

We use the [Meson](https://mesonbuild.com) build system to manage compilation
options, dependency tracking, and installation of the compiled package. Please
follow these steps below to build and install HDF5-UDF.

### Configuration

Meson uses a dedicated build directory. We create one named `build` and
proceed with the configuration using these commands:

```
$ mkdir -p build
$ meson -Dwith-python=true -Dwith-lua=true -Dwith-cpp=true . build
```

At least one of `-Dwith-python=true`, `-Dwith-lua=true`, or `-Dwith-cpp=true` options
must be set. It's still possible to build HDF5-UDF without any of these backends, but
that would be a useless outcome!

Support for sandboxing is strongly encouraged to be set, so it's enabled by default.
If you are conducting local tests and do not plan on reading datasets provided by
third-party, then it's possible to disable it with `-Dwith-sandbox=false`.

The installation prefix defaults to `/usr/local`, with the HDF5 I/O filter plugin
installed under `/usr/local/hdf5/lib/plugin`. The prefix can be changed by adding
`--prefix=/path/to/prefix` to the command line options. The plugin path can be set
with `--libexecdir=subdirectory/of/prefix`.

In order to use an alternative compiler, please set the `CXX` environment variable
so it points to that executable (e.g., `CXX=clang++`).

### Building

Simply run `ninja -C build` to compile the source code with the previously configured
options.

### Installing

Run `ninja -C build install` to effectively install HDF5-UDF on your filesystem.
