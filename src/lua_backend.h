/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: lua_backend.h
 *
 * Interfaces for Lua code parser and bytecode generation/execution.
 */
#ifndef __lua_backend_h
#define __lua_backend_h

#include "backend.h"

class LuaBackend : public Backend {
public:
    // Backend name
    std::string name();

    // Extension managed by this backend
    std::string extension();

    // Compile an input file into executable form
    std::string compile(
        std::string udf_file,
        std::string template_file,
        std::string compound_declarations,
        std::string &sourcecode,
        std::vector<DatasetInfo> &input_datasets);

    // Execute a user-defined-function
    bool run(
        const std::string filterpath,
        const std::vector<DatasetInfo> &input_datasets,
        const DatasetInfo &output_dataset,
        const char *output_cast_datatype,
        const char *udf_blob,
        size_t udf_blob_size);

    // Scan the UDF file for references to HDF5 dataset names.
    // We use this to store the UDF dependencies in the JSON payload.
    std::vector<std::string> udfDatasetNames(std::string udf_file);

    // Create a textual declaration of a struct given a compound map
    std::string compoundToStruct(const DatasetInfo &info, bool hardcoded_name);

private:
    std::string bytecode;
};

#endif /* __lua_backend_h */