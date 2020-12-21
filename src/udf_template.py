#
# HDF5-UDF: User-Defined Functions for HDF5
#
# File: udf_template.py
#
# HDF5 filter callbacks and main interface with the Python API.
#

# Note: python_backend.cpp loads the "cffi" module for us. By delegating that
# task to the C code we can keep strict sandboxing rules, as the import process
# requires access to the filesystem (i.e., readdir, stat, open, etc)

class PythonLib:
    def load(self, filterpath):
        # self.cffi is initialized from C code
        # self.ffi = cffi.FFI()

        self.ffi.cdef("""
            void       *pythonGetData(const char *);
            const char *pythonGetType(const char *);
            const char *pythonGetCast(const char *);
            const char *pythonGetDims(const char *);
            // compound_declarations_placeholder
            """, packed=True)

        # self.filterlib is initialized from C code
        # self.filterlib = self.ffi.dlopen(filterpath)

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
