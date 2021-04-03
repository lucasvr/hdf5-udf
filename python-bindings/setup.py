#!/usr/bin/env python3

from setuptools import setup, find_packages

long_description = """
This module provides a Python interface to HDF5-UDF so that users can
embed routines on HDF5 files programmatically.
"""

setup(
    name="hdf5_udf",
    version="1.0",
    description="User-defined functions for HDF5 - Python bindings",
    long_description=long_description,
    author="Lucas C. Villa Real",

    py_modules=["hdf5_udf"],
    python_requires=">=3.6",
    setup_requires=["cffi>=1.0.0"],
    install_requires=[
        "cffi>=1.0.0",
        "jsonschema>=3.2.0"
    ],

    cffi_modules=["build_ext.py:ffibuilder"],
    zip_safe=False,
)