#!/usr/bin/env python3

import os
import re
import glob
import cffi
import shutil
import pathlib

libname = "_pyhdf5_udf"

def clean():
    """ Remove temporarily built objects """
    for pattern in ("*.o", f"{libname}.c", f"{libname}.h", "*.so"):
        for f in glob.glob(pattern):
            os.unlink(f)
    for pattern in ("build", "dist", "__pycache__", "*.egg-info"):
        for d in glob.glob(pattern):
            shutil.rmtree(d)

def build(ffibuilder):
    """ Build the Python bindings """
    this_dir = pathlib.Path().resolve()
    h_file_name = f"{this_dir}/../src/hdf5-udf.h"
    with open(h_file_name) as h_file:
        # Remove preprocessor directives
        lns = h_file.read().splitlines()
        flt = filter(lambda ln: not re.match(r" *#", ln), lns)
        flt = map(lambda ln: ln.replace('extern "C" {', ''), flt)
        flt = map(lambda ln: ln.replace('}', ''), flt)
        ffibuilder.cdef(str("\n").join(flt))

    with open(f"{libname}.h", "w") as f:
        flt = map(lambda ln: ln.replace('extern "C" {', ''), lns)
        flt = map(lambda ln: ln.replace('}', ''), flt)
        f.write(str("\n").join(flt))

    ffibuilder.set_source(
        libname,
        f'#include "{libname}.h"',
        libraries=["hdf5-udf"],
        library_dirs=[f"{this_dir}/../build"],
        include_dirs=[this_dir],
        extra_link_args=["-Wl,-rpath,."],
    )
    ffibuilder.compile()

ffibuilder = cffi.FFI()
clean()
build(ffibuilder)