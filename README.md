# HDF5-UDF

HDF5-UDF is a mechanism to generate HDF5 dataset values on-the-fly using
user-defined functions (UDFs). The platform supports UDFs written in Python,
C++, and Lua under Linux, macOS, and Windows.
[Python bindings](https://hdf5-udf.readthedocs.io/en/latest) are provided
for those wishing to skip the command-line utility and programmatically embed
UDFs on their datasets.

# Table of Contents

1. [Overview](#overview)
    - [CPython backend](#cpython-backend)
    - [C++ backend](#c++-backend)
    - [LuaJIT backend](#luajit-backend)
2. [User-Defined Functions API](#user-defined-functions-api)
    - [Supported datatypes](#supported-datatypes)
    - [HDF5 groups](#hdf5-groups)
    - [Strings and compounds](#strings-and-compounds)
3. [Examples](#examples)
    - [A User-Defined Function in Python](#python-api)
    - [A User-Defined Function in C++](#c++-api)
    - [A User-Defined Function in Lua](#lua-api)
3. [Configuration notes](#configuration-notes)

# Overview

HDF5-UDF provides an interface that takes a piece of code provided by the user
and compiles it into a bytecode (or shared library) form. The resulting object
is saved into HDF5 as a binary blob. Each time that dataset is read by an
application, the HDF5-UDF I/O filter retrieves that blob from the dataset,
loads it into memory and executes the user-defined function -- which populates
the dataset values on-the-fly. There is no difference between accessing a dataset
whose values have been dynamically generated and a regular one. Both are retrieved
using the existing HDF5 API.

![](images/hdf5-udf.png)

The blob data saved into the dataset is accompanied by a piece of
[metadata](https://hdf5-udf.readthedocs.io/en/latest/#json-schema-for-hdf5-udf-datasets)
that helps HDF5-UDF parse the bytecode. That metadata includes the data type to
be produced, the output dataset name, its dimensions, dependencies, and the UDF
creator's digital signature (which allow users to [restrict the system resources
an UDF can access](https://hdf5-udf.readthedocs.io/en/latest/#trust-profiles)):

```
metadata = {
  "backend": "CPython",
  "bytecode_size": 879,
  "input_datasets": ["A", "B"],
  "output_dataset": "C",
  "output_datatype": "float",
  "output_resolution": [1024, 768],
  "signature": {
    "email": "lucasvr@Pagefault",
    "name": "Lucas C. Villa Real",
    "public_key": "17ryiejfF2SNlT+MCIaPLb7bKqo2FTX/PIiCDrh45Fc="
  }
}
```

## CPython backend

UDFs written in Python are compiled into bytecode form by the CPython interpreter.
All I/O between HDF5 and HDF5-UDF goes through Foreign Function Interfaces (FFIs),
meaning that there is no measurable overhead when accessing input and output datasets.
This backend requires Python 3.

## C++ backend

The C++ backend compiles the provided UDF using GCC (`g++`) or, if available,
Clang (`clang++`), into a shared library. The result is compressed and embedded
into HDF5.

## LuaJIT backend

UDFs written in Lua are processed by the LuaJIT interpreter. The output bytecode
is saved into HDF5 and interpreted by LuaJIT's just-in-time compiler when the
dataset is read by the application.

# User-Defined Functions API

The Lua, C++, and Python APIs are identical and provide the following simple
functions to interface with HDF5 datasets:

- `lib.getData("DatasetName")`: fetches DatasetName from the HDF5
   file and loads it into memory
- `lib.getDims("DatasetName")`: number of dimensions in DatasetName
   and their sizes
- `lib.getType("DatasetName")`: dataset type of DatasetName. See
   below for a list of supported dataset types
- `lib.string(member)`: get the value of a string datatype
- `lib.setString(member, value)`: write the given value to a string datatype.
   This API does boundary checks to prevent buffer overflows

The user-provided function must be named `dynamic_dataset`. That
function takes no input and produces no output; data exchange is
performed by reading from and writing to the datasets retrieved
by the API above. See the next section for examples on how to
get started with UDF scripts.

## Supported datatypes

The following data types are supported by UDF datasets:

- `int8`, `uint8` (`H5T_STD_I8LE`, `H5T_STD_U8LE`)
- `int16`, `uint16` (`H5T_STD_I16LE`, `H5T_STD_U16LE`)
- `int32`, `uint32` (`H5T_STD_I32LE`, `H5T_STD_U32LE`)
- `int64`, `uint64` (`H5T_STD_I64LE`, `H5T_STD_U64LE`)
- `float`, `double` (`H5T_IEEE_F32LE`, `H5T_IEEE_F64LE`)
- `string` (`H5T_C_S1`)
- `compound` (`H5T_COMPOUND`)

## HDF5 groups

HDF5-UDF also supports datasets stored in a non-flat hierarchy. The API accepts
dataset names that are prefixed by existing group names, as in
`lib.getData("/group/name/dataset")`. It is also possible to store a UDF dataset
on a given group by using the same syntax on the command line, such as 
`/group/name/dataset:resolution:datatype`.

## Strings and compounds

It is possible to write UDFs that take input from compounds and from strings (both
fixed- and variable-sized ones). `hdf5-udf` will print the name and layout of the
generated structure that you can use to iterate over the input data members. Please
refer to the [examples](https://github.com/lucasvr/hdf5-udf/tree/master/examples)
directory for a guidance on how to access such datatypes from Python, C/C++ and Lua.

Also, one can write UDFs that output such datatypes. Strings have a default fixed
size of 32 characters; that value can be changed by the user using the `(N)` modifier.
For instance, to output strings with at most 8 characters one can declare a dataset
like `dataset_name:resolution:string(8)`.

The syntax for outputting compounds is slightly different, as members may have
distinct datatypes: `dataset_name:{member:type[,member:type...]}:resolution`.
A sample compound named "placemark" with a single "location" member can be entered
as `placemark:{location:string}:1000` to `hdf5-udf`.

Multiple compound members must be
separated by a comma within the curly braces delimiters. Note that it is possible
to use the `(N)` modifier with string members that belong to a compound too. In the
previous example, one could write `placemark:{location:string(48)}:1000` to limit
location strings to 48 characters.

# Examples

The following simple examples should get you started into HDF5-UDF. A more
comprehensive set of examples is provided at the
[user-defined functions repository](https://github.com/lucasvr/user-defined-functions).

Also, make sure to read the template files for
[Lua](https://github.com/lucasvr/hdf5-udf/blob/master/src/udf_template.lua),
[Python](https://github.com/lucasvr/hdf5-udf/blob/master/src/udf_template.py), and
[C++](https://github.com/lucasvr/hdf5-udf/blob/master/src/udf_template.cpp)
to learn more about the APIs behind the HDF5-UDF `lib` interface.

## Python API

In this example, dataset values for "C" are computed on-the-fly as the sum
of compound dataset members "A.foo" and "B.bar".

```
def dynamic_dataset():
    a_data = lib.getData("A")
    b_data = lib.getData("B")
    c_data = lib.getData("C")
    n = lib.getDims("C")[0] * lib.getDims("C")[1]

    for i in range(n):
        c_data[i] = a_data[i].foo + b_data[i].bar
```

## C++ API

This example shows dataset values for "C" being generated on-the-fly as the sum
of datasets "B" and "C". Note that the entry point is marked `extern "C"` and that
calls to `lib.getData()` explicitly state the storage data type.

```
extern "C" void dynamic_dataset()
{
    auto a_data = lib.getData<int>("A");
    auto b_data = lib.getData<int>("B");
    auto c_data = lib.getData<int>("C");
    auto n = lib.getDims("C")[0] * lib.getDims("C")[1];

    for (size_t i=0; i<n; ++i)
        c_data[i] = a_data[i] + b_data[i];
}
```

## Lua API

In this example, dataset values for "C" are computed on-the-fly as the sum
of datasets "A" and "B". The UDF is written in Lua.

```
function dynamic_dataset()
    local a_data = lib.getData("A")
    local b_data = lib.getData("B")
    local c_data = lib.getData("C")
    local n = lib.getDims("C")[1] * lib.getDims("C")[2]
    for i=1, n do
        c_data[i] = a_data[i] + b_data[i]
    end
end
```

# Configuration notes

If the program has been installed to a directory other than `/usr/local`, then
make sure to configure the HDF5 filter search path accordingly:

```
$ export HDF5_PLUGIN_PATH=/installation/path/hdf5/lib/plugin
```

The main program takes as input a few required arguments: the HDF5 file, the
user-defined script, and the output dataset name/resolution/data type. If
we were to create a `float` dataset named "temperature" with 1000x800 cells
(and whose script is named "udf.py") then the following command would do
it (while appending the result to "myfile.h5"):

```
$ hdf5-udf myfile.h5 udf.py temperature:1000x800:float
```

It is also possible to let the main program infer the output dataset information
based on the UDF script -- as long as the script takes input from at least one
existing dataset. In that case, if the dataset name/resolution/data type is
omitted from the command line, the main program:

1. Identifies calls to `lib.getData()` in the UDF script and checks if the dataset
   name given as argument to that function exists in the HDF5 file. If it doesn't,
   then that name is used for the output dataset.
2. Identifies the resolution and data types of existing datasets taken as input.
   If all input datasets have the same resolution and data type, then the output
   dataset is produced with the same characteristics.

In such cases, invoking the program is as simple as:

```
$ hdf5-udf myfile.h5 udf.py
```

It is also possible to have more than one dataset produced by a single script.
In that case, information regarding each output variable can be provided in the
command line as extra arguments to the main program. Alternatively, their names,
resolution and data types can be guessed from the UDF script as mentioned before.
