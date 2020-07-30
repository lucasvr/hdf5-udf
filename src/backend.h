/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: backend.h
 *
 * Interfaces with supported code parsers and generators.
 */
#ifndef __backend_h
#define __backend_h

#include <stdbool.h>
#include <vector>
#include <string>
#include "dataset.h"

class Backend {
public:
    // Backend name (e.g., "LuaJIT")
    virtual std::string name() {
        return "";
    }

    // Backend file extension (e.g., ".lua")
    virtual std::string extension() {
        return "";
    }

    // Compile an input file into executable form
    virtual std::string compile(std::string udf_file, std::string template_file) {
        return "";
    }

    // Execute a user-defined-function
    virtual bool run(
        const std::string filterpath,
        const std::vector<DatasetInfo> input_datasets,
        const DatasetInfo output_dataset,
        const char *output_cast_datatype,
        const char *udf_blob,
        size_t udf_blob_size)
    {
        return false;
    }

    // Scan the UDF file for references to HDF5 dataset names.
    // We use this to store the UDF dependencies in the JSON payload.
    virtual std::vector<std::string> udfDatasetNames(std::string udf_file) {
        return std::vector<std::string>();
    }
};

// Get a backend by their name (e.g., "LuaJIT")
Backend *getBackendByName(std::string name);

// Get a backend by file extension (e.g., ".lua")
Backend *getBackendByFileExtension(std::string name);

#endif /* __backend_h */