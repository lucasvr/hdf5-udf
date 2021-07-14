.. HDF5-UDF documentation master file, created by
   sphinx-quickstart on Thu Apr 15 22:42:44 2021.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:


User-Defined functions for HDF5
...............................

HDF5-UDF is a mechanism to generate HDF5 dataset values on-the-fly using
user-defined functions (UDFs). The platform supports UDFs written in Python,
C++, and Lua under Linux, macOS, and Windows.

This page provides documentation for UDF writers and for those wishing to
programmatically compile and store UDFs through HDF5-UDF's C and Python APIs.


Installation
============

Please refer to the project page on `GitHub <https://github.com/lucasvr/hdf5-udf/blob/master/INSTALL.md>`_
for up-to-date instructions on how to install the HDF5-UDF library and its
utilities from binary packages or in source code form.


Once the main library has been installed on your system, PIP installs the
Python APIs needed to programmatically compile and attach UDFs to HDF5 files:

.. code-block::

   pip install pyhdf5-udf


Getting started
===============

We begin by writing the function that we will compile and store on HDF5. In the
example below, the dataset will be named "simple". Note the calls to ``lib.getData()``
and ``lib.getDims()``: they are part of HDF5-UDF runtime and provide direct access to
the dataset buffer and its dimensions, respectively. See
:ref:`The lib interface for UDF writers` for details on that interface. The entry
point of the UDF is always a function named ``dynamic_dataset``.

.. code-block::

   from hdf5_udf import UserDefinedFunction

   def dynamic_dataset():
      from math import sin
      data = lib.getData('simple')
      dims = lib.getDims('simple')
      for i in range(dims[0]):
         data[i] = sin(i)

Now, we use the ``inspect`` module to capture the function's source code and save
it as a file on disk. HDF5-UDF recognizes files ending on ``.py``, ``.cpp``, and
``.lua``.

.. code-block::

   import inspect
   with open("/tmp/udf.py", "w") as f:
      f.write(inspect.getsource(dynamic_dataset))

Last, we declare a ``UserDefinedFunction`` object and describe the dataset: its
name, type, and dimensions. Next, we compile it into a bytecode form and store it
in the provided HDF5 file.

.. code-block::

   with UserDefinedFunction(hdf5_file='/path/to/file.h5', udf_file='/tmp/udf.py') as udf:
      udf.push_dataset({'name': 'simple', 'datatype': 'float', 'resolution': [1000]})
      udf.compile()
      udf.store()

At this point the dataset has been created and it's possible to retrieve its data.
Note that the call to ``f['simple'][:]`` triggers the execution of the bytecode we
just compiled. There's more to the picture than meets the eye!

.. code-block::

   import h5py
   f = h5py.File('/path/to/file.h5')
   simple = f['simple][:]
   ...
   f.close()


The lib interface for UDF writers
=================================

HDF5-UDF comes with a few primary interfaces and helper methods to ease
storing and retrieving values of string-based datasets. It is not necessary to
explicitly instantiate a ``PythonLib`` object: you can simply use the ``lib``
one provided by HDF5-UDF as seen in the examples below.

.. module:: udf_template
.. autoclass:: PythonLib
   :members:
   :member-order: bysource


UDF compilation and storage interfaces
======================================

User-defined functions can be compiled and stored in HDF5 from the command-line
or from HDF5-UDF's APIs. To date, the project provides a
`low-level C API <https://github.com/lucasvr/hdf5-udf/blob/default/src/hdf5-udf.h>`_
and a Python abstraction of that API, documented below.


Command-line interface
----------------------

Given an UDF file with a supported backend (C++, Python, or Lua) and a target
HDF5 file, the ``hdf5-udf`` utility can be used to compile and store the UDF
on that file.

.. code-block::

    Syntax: hdf5-udf [flags] <hdf5_file> <udf_file> [udf_dataset..]

    Flags:
      --overwrite              Overwrite existing UDF dataset(s)
      --save-sourcecode        Include source code as metadata of the UDF

When populating values for a dataset with a native data type, the following
syntax applies to describe that dataset:

.. code-block::

    name:resolution:type

where:

* name: name of the UDF dataset
* resolution: dataset resolution, with dimensions separated by the ``x``
  character. Examples: ``XSize``, ``XSize``x``YSize``,
  ``XSize``x``YSize``x``ZSize``
* type: ``[u]int8``, ``[u]int16``, ``[u]int32``, ``[u]int64``, ``float``,
  ``double``, ``string``, or ``string(NN)``. If unset, strings have a fixed
  size of 32 characters.

When populating values for a dataset with a compound data type, the following
syntax is used instead:

.. code-block::

      name:{member:type[,member:type...]}:resolution


Python interface
----------------

It is also possible to bypass the command line and embed the instructions to
compile and store the UDF in a HDF5 file using HDF5-UDF's Python API. This API
is an abstraction of the `low-level C API <https://github.com/lucasvr/hdf5-udf/blob/default/src/hdf5-udf.h>`_.

.. module:: hdf5_udf
.. autoclass:: UserDefinedFunction
   :members:
   :member-order: bysource


JSON schema for HDF5-UDF datasets
---------------------------------

The dictionary used to describe the UDF dataset(s) must follow the schema
below.

.. jsonschema:: ../hdf5_udf_resources/hdf5_udf-schema.json

.. include:: security.rst

.. include:: settings.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
