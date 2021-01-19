/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: cpp_backend.h
 *
 * Interfaces for C++ code parser and shared library generation/execution.
 */
#ifndef __cpp_backend_h
#define __cpp_backend_h

#include "backend.h"

class CppBackend : public Backend {
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
    // Compress a data buffer
    std::string compressBuffer(const char *data, size_t usize);

    // Decompress a data buffer
    std::string decompressBuffer(const char *data, size_t csize);
};

#endif /* __cpp_backend_h */
