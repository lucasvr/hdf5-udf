#!/usr/bin/env python3

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requires = f.read().splitlines()

setup(
    name="PyHDF5-UDF",
    version="1.3",
    description="User-defined functions for HDF5 - Python bindings",
    description_content_type="text/x-rst",
    long_description=open("README.txt").read(),
    long_description_content_type="text/x-rst",
    author="Lucas C. Villa Real",
    author_email="lucasvr@gobolinux.org",
    url="https://hdf5-udf.readthedocs.io",

    py_modules=["hdf5_udf"],
    python_requires=">=3.5.10",
    setup_requires=[
        "wheel",
        "cffi>=1.0.0"
    ],
    install_requires=requires,
    packages=find_packages(),
    include_package_data=True,

    cffi_modules=["build_ext.py:ffibuilder"],
    zip_safe=False,
)
