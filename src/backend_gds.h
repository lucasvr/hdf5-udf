/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: backend_gds.h
 *
 * NVIDIA GPUDirect Storage backend definitions.
 */
#ifndef __backend_gds_h
#define __backend_gds_h

#include <cufile.h>
#include "backend.h"

// DeviceMemory: allocate and register memory on the device for GPUDirect I/O
struct DeviceMemory {
    DeviceMemory(size_t alloc_size);
    ~DeviceMemory();
    void clear();

    void *dev_mem;
    size_t size;
};

// DirectStorage: simple interface to register and deregister the GDS driver.
struct DirectStorage {
    DirectStorage();
    ~DirectStorage();
};

// DirectFile: open and configures a HDF5 file for Direct I/O
struct DirectFile {
    DirectFile();
    ~DirectFile();
    bool open(std::string hdf5_file);
    void close();

    int file_fd;
    hid_t file_id;
    CUfileDescr_t gds_descr;
    CUfileHandle_t gds_handle;

};

// DirectDataset: open a dataset
struct DirectDataset {
    static bool read(hid_t dset_id, const CUfileHandle_t &gds_handle, DeviceMemory &mm);
    static bool copyToHost(DeviceMemory &mm, void **host_mem);
};

class GDSBackend : public Backend {
public:
    // Backend name
    std::string name();

    // Extension managed by this backend
    std::string extension();

    // Compile an input file into executable form
    std::string compile(
        std::string udf_file,
        std::string compound_declarations,
        std::string &source_code,
        std::vector<DatasetInfo> &input_datasets);

    // Execute a user-defined-function
    bool run(
        const std::string libpath,
        const std::vector<DatasetInfo> &input_datasets,
        const DatasetInfo &output_dataset,
        const char *output_cast_datatype,
        const char *udf_blob,
        size_t udf_blob_size,
        const json &rules);

    // Scan the UDF file for references to HDF5 dataset names.
    // We use this to store the UDF dependencies in the JSON payload.
    std::vector<std::string> udfDatasetNames(std::string udf_file);

    // Create a textual declaration of a struct given a compound map
    std::string compoundToStruct(const DatasetInfo &info, bool hardcoded_name);

    // Allocate memory for an input or scratch dataset in device memory
    void *alloc(size_t size);

    // Free memory previously allocated for an input or scratch dataset
    void free(void *dev_mem);

    // Zeroes out a range of memory previously allocated for an input or scratch dataset
    void clear(void *dev_mem, size_t size);

    // Copy data from device memory to a newly allocated memory chunk in the host
    void *deviceToHost(void *dev_mem, size_t size);

    // Get a reference to the memory handler of the given device memory address
    DeviceMemory *memoryHandler(void *dev_mem);

private:
    std::map<void*, DeviceMemory*> memory_map;
};

#endif /* __backend_gds_h */