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
    def load(self, libpath):
        # self.cffi is initialized from C code
        # self.ffi = cffi.FFI()

        self.ffi.cdef("""
            void       *pythonGetData(const char *);
            const char *pythonGetType(const char *);
            const char *pythonGetCast(const char *);
            const char *pythonGetDims(const char *);
            // compound_declarations_placeholder
            """, packed=True)

        # self.udflib is initialized from C code
        # self.udflib = self.ffi.dlopen(libpath)

    def string(self, structure):
        # Strings are embedded in a structure with a single
        # 'value' member.
        if hasattr(structure, "value"):
            return self.ffi.string(structure.value).decode("utf-8")
        # The user may also provide a direct pointer to the
        # string member.
        return self.ffi.string(structure).decode("utf-8")

    def setString(self, structure, s):
        if hasattr(structure, "value"):
            n = len(s) if len(s) <= len(structure.value) else len(structure.value)
            structure.value[0:n] = s[0:n]
        else:
            n = len(s) if len(s) <= len(structure) else len(structure)
            structure[0:n] = s[0:n]

    def getData(self, name):
        name = self.ffi.new("char[]", name.encode("utf-8"))
        cast = self.udflib.pythonGetCast(name)
        data = self.udflib.pythonGetData(name)
        ctype = self.ffi.string(cast).decode("utf-8")
        return self.ffi.cast(ctype, data)

    def getType(self, name):
        name = self.ffi.new("char[]", name.encode("utf-8"))
        return self.ffi.string(self.udflib.pythonGetType(name))

    def getDims(self, name):
        name = self.ffi.new("char[]", name.encode("utf-8"))
        dims = self.udflib.pythonGetDims(name)
        dims = self.ffi.string(dims).decode("utf-8")
        return tuple([int(dim) for dim in dims.split("x")])

lib = PythonLib()

# User-Defined Function

# user_callback_placeholder
