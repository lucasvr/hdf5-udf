#
# HDF5-UDF: User-Defined Functions for HDF5
#
# File: udf_template.py
#
# HDF5 filter callbacks and main interface with the Python API.
#

import os
from cffi import FFI

class PythonLib:
    def load(self, filterpath):
        self.ffi = FFI()
        self.ffi.cdef("""
            void       *pythonGetData(const char *);
            const char *pythonGetType(const char *);
            const char *pythonGetCast(const char *);
            const char *pythonGetDims(const char *);
            """)
        self.filterlib = self.ffi.dlopen(filterpath)

    def getData(self, name):
        name = self.ffi.new("char[]", name.encode("utf-8"))
        cast = self.filterlib.pythonGetCast(name)
        data = self.filterlib.pythonGetData(name)
        ctype = self.ffi.string(cast).decode("utf-8")
        return self.ffi.cast(ctype, data)

    def getType(self, name):
        name = self.ffi.new("char[]", name.encode("utf-8"))
        return self.ffi.string(self.filterlib.pythonGetType(name))

    def getDims(self, name):
        name = self.ffi.new("char[]", name.encode("utf-8"))
        dims = self.filterlib.pythonGetDims(name)
        dims = self.ffi.string(dims).decode("utf-8")
        return tuple([int(dim) for dim in dims.split("x")])

lib = PythonLib()

# User-Defined Function

# user_callback_placeholder
