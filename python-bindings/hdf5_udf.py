#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""User-Defined Functions for HDF5: Python bindings

This module provides a Python interface to HDF5-UDF so that users can
embed routines on HDF5 files programmatically.

Please refer to https://github.com/lucasvr/hdf5-udf for more details
on this project.
"""

import json
from jsonschema import validate
from _pyhdf5_udf import lib, ffi

class UserDefinedFunction:
    """Store a user-defined function on a HDF5 file.

    Parameters
    ----------
    hdf5_file : str
        Path to existing HDF5 file
    udf_file : str
        Path to file implementing the user-defined function
    """
    def __init__(self, hdf5_file="", udf_file=""):
        self.ctx = lib.libudf_init(
            bytes(hdf5_file, 'utf-8'),
            bytes(udf_file, 'utf-8'))

    def __enter__(self):
        return self

    def __del__(self):
        self.destroy()

    def __exit__(self, exc_type, exc_value, traceback):
        self.destroy()

    def set_option(self, option="", value=""):
        """Set an option given by a key/value pair.

        Parameters
        ----------
        option : str
            Name of the option to configure. Recognized option names include:

            - "overwrite": Overwrite existing UDF dataset? (default: False)
            - "save_sourcecode": Save the source code as metadata? (default: False)
        value : str, bool
            Value to set `option` to.

        Raises
        ------
        TypeError
            If the given data type is not recognized

        Returns
        -------
        bool
            True if successful, False otherwise.
        """

        if type(option) not in [str, bytes]:
            raise TypeError("Unsupported data type: only str and bytes are allowed")
        elif type(value) not in [bool, str, bytes]:
            raise TypeError("Unsupported data type: only str, bytes, and bool are allowed")
        return lib.libudf_set_option(
            bytes(option, 'utf-8'),
            bytes(value, 'utf-8'),
            self.ctx)

    def push_dataset(self, description):
        """Define a new UserDefinedFunction dataset.

        Parameters
        ----------
        description : dict
            Describe the dataset: its name, data type, size, and members
            (if a compound data type). For native datasets the following
            keys are expected: ``name``, ``datatype``, ``resolution``.

            Compound datasets must provide an extra ``members`` key.
            Objects of the ``members`` array must include two properties:
            ``name`` and ``datatype``.

        Examples
        --------
        Dataset with a native data type:

        .. code-block::

            {"name": "MyDataset", "datatype": "int32", "resolution": [100,100]}

        Dataset with a compound data type:
        .. code-block::

            {
                "name": "MyCompoundDataset",
                "datatype": "compound",
                "resolution": 100,
                "members": [
                    {"name": "Identifier", "datatype": "int64"},
                    {"name": "Description", "datatype": "string(80)"}
                ]
            }

        Raises
        ------
        TypeError
            If `description` or its members hold an unexpected data type
        ValueError
            If description dictionary misses mandatory keys

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        # Validate dictionary keys and data types
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://github.com/lucasvr/hdf5-udf",
            "description": "JSON schema for HDF5-UDF datasets",
            "type": "object",
            "properties": {
                "name": {
                    "description": "Dataset name",
                    "type": "string"
                },
                "datatype": {
                    "description": "Data type",
                    "type": "string"
                },
                "resolution": {
                    "description": "Dataset dimensions",
                    "type": "array",
                    "items": {
                        "type": "number",
                        "exclusiveMinimum": 0
                    },
                    "minItems": 1
                },
                "members": {
                    "description": "Compound members",
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "description": "Compound member name",
                                "type": "string"
                            },
                            "datatype": {
                                "description": "Compound member data type",
                                "type": "string"
                            },
                        }
                    },
                    "minItems": 1
                }
            },
            "required": ["name", "datatype", "resolution"]
        }

        validate(instance=description, schema=schema)

        # Conversion of dictionary to the flat string format expected by libudf
        d = description
        resolution = "x".join([str(x) for x in d["resolution"]])
        if not "members" in description:
            desc = f"{d['name']}:{resolution}:{d['datatype']}"
        else:
            members = ""
            for element in d["members"]:
                members += "{}:{},".format(element["name"], element["datatype"])
            desc = f"{d['name']}:{{{members[:-1]}}}:{resolution}"
        import sys
        sys.stderr.write(f"{desc}\n")
        return lib.libudf_push_dataset(bytes(desc, 'utf-8'), self.ctx)

    def compile(self):
        """Compile the UserDefinedFunction into bytecode form.
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        return lib.libudf_compile(self.ctx)

    def store(self):
        """Store the compiled bytecode on the HDF5 file.
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        return lib.libudf_store(self.ctx)

    def destroy(self):
        """Release resources allocated for the object.

        This function must be called to ensure handles are closed and avoid
        resource leaks. It is best, however, to use simply use a context
        manager to define `UserDefinedFunction` objects::

            with UserDefinedFunction(hdf5_file='file.h5', udf_file='udf.py') as udf:
                udf.push_dataset(...)
                udf.compile()
                udf.store()
        """
        lib.libudf_destroy(self.ctx)
        self.ctx = ffi.NULL
