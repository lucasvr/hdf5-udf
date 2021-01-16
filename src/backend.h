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

struct AssembleData {
    std::string udf_file;                 // UDF file
    std::string template_file;            // backend-specific template file

    std::string compound_placeholder;     // placeholder string in template file
    std::string compound_decl;            // declaration of auto-generated C structures

    std::string methods_decl_placeholder; // placeholder string in template file
    std::string methods_decl;             // prototype of auxiliary methods
    std::string methods_impl_placeholder; // placeholder string in template file
    std::string methods_impl;             // actual implementation of those methods

    std::string callback_placeholder;     // UDF placeholder string in template file
    std::string extension;                // file name extension to use
};

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
    virtual std::string compile(
        std::string udf_file,
        std::string template_file,
        std::string compound_declarations,
        std::vector<DatasetInfo> &datasets) {
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

    // Create a textual declaration of a struct given a compound map
    virtual std::string compoundToStruct(const DatasetInfo info, bool hardcoded_name) {
        return std::string("");
    }

    // Helper function: combine the UDF template file and the user-defined-function
    // file into one, saving the result to a temporary file on disk that ends on the
    // on the provided extension. The user-defined-function is injected in the template
    // file right where the placeholder string is found.
    std::string assembleUDF(const AssembleData &data);

    // Helper function: save a data blob to a temporary file on disk whose name ends
    // on the given extension.
    std::string writeToDisk(const char *data, size_t size, std::string extension);

    // Helper function: converts a string into a valid C variable name
    std::string sanitizedName(std::string name);
};

// Get a backend by their name (e.g., "LuaJIT")
Backend *getBackendByName(std::string name);

// Get a backend by file extension (e.g., ".lua")
Backend *getBackendByFileExtension(std::string name);

#endif /* __backend_h */