/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: PythonBackend.h
 *
 * Interfaces for Python code parser and bytecode generation/execution.
 */
#ifndef __python_backend_h
#define __python_backend_h

#include <Python.h>
#include <marshal.h>
#include "backend.h"

class PythonBackend : public Backend {
public:
    // Backend name
    std::string name();

    // Extension managed by this backend
    std::string extension();

    // Compile an input file into executable form
    std::string compile(std::string udf_file, std::string template_file);

    // Execute a user-defined-function
    bool run(
        const std::string filterpath,
        const std::vector<DatasetInfo> input_datasets,
        const DatasetInfo output_dataset,
        const char *output_cast_datatype,
        const char *udf_blob,
        size_t udf_blob_size);

    // Scan the UDF file for references to HDF5 dataset names.
    // We use this to store the UDF dependencies in the JSON payload.
    std::vector<std::string> udfDatasetNames(std::string udf_file);

private:
    void printPyObject(PyObject *obj);
};

#endif /* __python_backend_h */
