.. HDF5-UDF documentation master file, created by
   sphinx-quickstart on Thu Apr 15 22:42:44 2021.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:


User-Defined functions for HDF5
===============================

Installing the main library
---------------------------

Please refer to the project page on `GitHub <https://github.com/lucasvr/hdf5-udf/blob/master/INSTALL.md>`_
for up-to-date instructions on how to install the HDF5-UDF library and its
utilities from binary packages or in source code form.


Installing the Python bindings
------------------------------

Once the main library has been installed on your system, PIP will bring you the
latest and greatest:

.. code-block::

   pip install pyhdf5-udf

Getting started
----------------

We begin by writing the function that we will compile and store on HDF5. In the
example below, the dataset will be named "simple". Note the calls to ``lib.getData()``
and ``lib.getDims()``: they are part of HDF5-UDF runtime and provide direct access to
the dataset buffer and its dimensions, respectively. See
:ref:`The lib interface available to UDFs` for details on that interface.

.. code-block::

   from hdf5_udf import UserDefinedFunction

   def dynamic_dataset():
      from math import sin
      data = lib.getData('simple')
      dims = lib.getDims('simple')
      for i in range(dims[0]):
         data[i] = sin(i)

Now, we use the ``inspect`` module to capture the function's source code and save
it as a file on disk. HDF5-UDF understands files ending on ``.py``, ``.cpp``, and
``.lua``.

.. code-block::

   import inspect
   with open("/tmp/udf.py", "w") as f:
      f.write(inspect.getsource(dynamic_dataset))

Last, we declare a ``UserDefinedFunction`` object and describe the dataset: its
name, type, and dimensions. Next, we compile it into a bytecode form and store it
on the provided HDF5 file.

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


The lib interface available to UDFs
-----------------------------------

The lib interface is automatically generated based on the previously available
datasets of the HDF5 file the UDF is built and attached to. The API comes with
three primary interfaces and a couple of helper methods to ease storage and
retrieval of string members.

- ``lib.getData("DatasetName")``: fetches DatasetName from the HDF5 file and loads it into memory
- ``lib.getDims("DatasetName")``: number of dimensions in DatasetName and their sizes
- ``lib.getType("DatasetName")``: dataset type of DatasetName. See below for a list of supported dataset types
- ``lib.string(member)``: get the value of a string datatype
- ``lib.setString(member, value)``: write the given value to a string datatype. This API does boundary checks to prevent buffer overflows


API documentation
-----------------

.. module:: hdf5_udf
.. autoclass:: UserDefinedFunction
   :members:
   :member-order: bysource

JSON schema for HDF5-UDF datasets
=================================

.. jsonschema:: ../hdf5_udf_resources/hdf5_udf-schema.json

.. include:: security.rst

.. include:: settings.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
