/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: backend_cuda.h
 *
 * CUDA backend definitions.
 */
#ifndef __backend_cuda_h
#define __backend_cuda_h

#include <hdf5.h>
#include <tuple>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvcomp/snappy.hpp>
#include <nvcomp.hpp>
#include "backend.h"

#define cudaCheckError() do { \
 cudaError_t e = cudaGetLastError(); \
 if (e!=cudaSuccess) { \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
   exit(0); \
 } \
} while(0)

#define FAIL(msg...) do { \
    fprintf(stderr, msg); \
    return false; \
} while (0)

// DeviceMemory: allocate and register memory on the device
struct DeviceMemory {
    DeviceMemory(const DeviceMemory *src, size_t offset, size_t alloc_size);
    DeviceMemory(size_t alloc_size);
    ~DeviceMemory();
    void clear();

    void *dev_mem;
    size_t size;
    bool is_borrowed_mem;
};

// DmaFile: interfaces to register file data transfers via DMA.
// Not used by this backend.
struct DmaFile {
    DmaFile();
    ~DmaFile();

    bool open(std::string hdf5_file, bool warn_on_error);
    void close();

    int file_fd;
    hid_t file_id;
    hid_t fapl_id;
};

// DirectDataset: read a dataset to device memory
struct DirectDataset {
    static bool read(hid_t dset_id, hid_t hdf5_datatype, DmaFile *DmaFile, DeviceMemory &mm);
    static bool copyToHost(DeviceMemory &mm, void **host_mem);
    static bool copyToDevice(const void *host_mem, hsize_t size, DeviceMemory &mm);
};

class CudaBackend : public Backend {
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

#endif /* __backend_cuda_h */
