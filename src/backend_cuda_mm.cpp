/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: backend_cuda_mm.cpp
 *
 * CUDA backend helper classes.
 */
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <hdf5.h>
#include <H5FDdirect.h>
#include <H5FDdevelop.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <omp.h>
#include <H5FDgds.h>
#include <nvcomp/snappy.hpp>
#include <nvcomp.hpp>

#include <string>
#include <tuple>
#include <vector>
#include "backend_cuda.h"
#include "dataset.h"
#include "debug.h"
#include "os.h"

#define cudaCheckError() do { \
 cudaError_t e = cudaGetLastError(); \
 if (e!=cudaSuccess) { \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
   exit(0); \
 } \
} while(0)

#define MB(n) (1024*1024*(n))
#define ALIGN_1MB(size) ((size + (0x100000-1)) & ~(0x100000-1))

#define FAIL(msg...) do { \
    fprintf(stderr, msg); \
    return false; \
} while (0)

#ifndef MAX
# define MAX(x,y) (x) > (y) ? (x) : (y)
#endif

// DeviceMemory: allocate and register memory on the device for GPUDirect I/O
DeviceMemory::DeviceMemory(size_t alloc_size, cudaStream_t stream) :
    dev_mem(NULL),
    size(alloc_size),
    total_size(ALIGN_1MB(alloc_size)),
    is_borrowed_mem(false)
{
    cudaSetDevice(0);
    cudaCheckError();

    // TODO: replace the following with a call to H5FDalloc_mem(). The GDS
    // file driver will allocate device memory and register the buffer for us.
    cudaError_t err = cudaMallocAsync(&dev_mem, total_size, stream);
    if (err != cudaSuccess)
    {
        const char *errmsg = cudaGetErrorString(err);
        fprintf(stderr, "Failed to allocate %ld page-locked bytes on GPU memory: %s\n",
                size, errmsg);
        dev_mem = NULL;
        return;
    }

    CUfileError_t gds_err = cuFileBufRegister(dev_mem, total_size, 0);
    if (gds_err.err != CU_FILE_SUCCESS)
    {
        fprintf(stderr, "Failed to register buffer size %#lx: %s\n", total_size, CUFILE_ERRSTR(gds_err.err));
        cudaFree(dev_mem);
        dev_mem = NULL;
    }
}

DeviceMemory::DeviceMemory(const DeviceMemory *src, size_t offset, size_t alloc_size) :
    dev_mem((void *)(((char *) src->dev_mem) + offset)),
    size(alloc_size),
    total_size(ALIGN_1MB(alloc_size)),
    is_borrowed_mem(true)
{
}

DeviceMemory::~DeviceMemory()
{
    if (dev_mem && !is_borrowed_mem)
    {
        // TODO: replace the following with a call to H5FDfree_mem(). The GDS
        // file driver will deregister the buffer and deallocate memory for us.
        cuFileBufDeregister(dev_mem);
        cudaError_t err = cudaFree(dev_mem);
        if (err != cudaSuccess)
        {
            const char *errmsg = cudaGetErrorString(err);
            fprintf(stderr, "Failed to free memory at %p: %s\n", dev_mem, errmsg);
        }
    }
}

void DeviceMemory::clear()
{
    cudaMemset(dev_mem, 0, size);
}

// DirectFile: open and configures a HDF5 file for Direct I/O
DirectFile::DirectFile() : file_id(-1), file_driver(NULL)
{
}

DirectFile::~DirectFile()
{
    this->close();
}

bool DirectFile::open(std::string hdf5_file, bool warn_on_error=true)
{
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    if (fapl_id < 0)
        FAIL("Failed to create HDF5 file access property list\n");

    // Open file with the NVIDIA GPUDirect Storage VFD.
    // We use default values for boundary, block size, and copy buffer size.
    size_t boundary = 0, block_size = 0, copy_buffer_size = 0;
    if (H5Pset_fapl_gds(fapl_id, boundary, block_size, copy_buffer_size) < 0)
        FAIL("Failed to enable the NVIDIA GPUDirect Storage VFD driver\n");

    file_id = H5Fopen(hdf5_file.c_str(), H5F_ACC_RDONLY, fapl_id);
    if (file_id < 0)
    {
        if (warn_on_error)
            FAIL("Failed to open %s with GPUDirect Storage\n", hdf5_file.c_str());
        return false;
    }

    // Resort to the new H5Fget_file_driver_handle() API to borrow a reference to the
    // low-level file driver handle. The obtained handle can be provided on calls to
    // the H5FD* APIs. The handle becomes invalid upon a call to H5Fclose(file_id).
    if (H5Fget_file_driver_handle(file_id, (void **) &file_driver) < 0)
        FAIL("Failed to get file driver handle\n");

    return true;
}

void DirectFile::close()
{
    if (file_id >= 0)
    {
        H5Fclose(file_id);
        file_id = -1;
    }
    if (fapl_id >= 0)
    {
        H5Pclose(fapl_id);
        fapl_id = -1;
    }

    file_driver = NULL;
}

// DirectDataset: routines to read a dataset using GPUDirect Storage I/O
bool DirectDataset::read(hid_t dset_id, DirectFile *directfile, DeviceMemory &mm)
{
    // TODO: this implementation only supports delegating workload to a single GPU
    cudaSetDevice(0);
    cudaCheckError();

    auto create_plist_id = H5Dget_create_plist(dset_id);
    auto dset_layout = H5Pget_layout(create_plist_id);
    bool ret = false;

    if (dset_layout == H5D_CHUNKED)
        ret = readChunked(dset_id, directfile, mm);
    else if (dset_layout == H5D_CONTIGUOUS)
        ret = readContiguous(dset_id, directfile, mm);

    H5Pclose(create_plist_id);
    return ret;
}

bool DirectDataset::parseChunks(
    hid_t dset_id,
    hid_t fspace_id,
    std::vector<hsize_t> &dims,
    std::vector<hsize_t> &cdims,
    std::vector<std::tuple<haddr_t, haddr_t, hsize_t, hsize_t>> &data_blocks)
{
    // Size of each grid element, in bytes
    auto datatype = H5Dget_type(dset_id);
    auto dset_element_size = getStorageSize(datatype);
    H5Tclose(datatype);
    if (dset_element_size < 0)
        FAIL("Unsupported datatype\n");

    // Get number of chunks in this dataset
    if (H5Sselect_all(fspace_id) < 0)
        FAIL("Failed to select the full extent of the dataset\n");

    hsize_t nchunks = 0;
    if (H5Dget_num_chunks(dset_id, fspace_id, &nchunks) < 0)
        FAIL("Failed to retrieve the number of chunks in the dataset\n");

    // Get dataset full grid size
    int ndims = H5Sget_simple_extent_ndims(fspace_id);
    if (ndims < 0)
        FAIL("Failed to retrieve the dimensionality of the dataspace\n");

    hsize_t dataset_num_elements = 1;
    dims.resize(ndims);
    H5Sget_simple_extent_dims(fspace_id, dims.data(), NULL);
    for (int i=0; i<ndims; ++i)
        dataset_num_elements *= dims[i];

    // Get chunk grid size
    hid_t create_plist_id = H5Dget_create_plist(dset_id);
    hsize_t chunk_num_elements = 1;
    cdims.resize(ndims);
    if (H5Pget_chunk(create_plist_id, ndims, cdims.data()) < 0)
    {
        H5Pclose(create_plist_id);
        FAIL("Failed to retrieve the size of chunks for the raw data\n");
    }
    H5Pclose(create_plist_id);
    for (int i=0; i<ndims; ++i)
        chunk_num_elements *= cdims[i];

    // Find out the configuration of the chunks in this dataset
    haddr_t mem_offset = 0;
    for (hsize_t i=0; i<nchunks; ++i)
    {
        haddr_t addr = 0;
        hsize_t offset[ndims], compressed_size = 0;
        if (H5Dget_chunk_info(dset_id, fspace_id, i, offset, NULL, &addr, &compressed_size) < 0)
            FAIL("Failed to retrieve information from chunk %ld\n", i);
        if (H5Dget_chunk_storage_size(dset_id, offset, &compressed_size) < 0)
            FAIL("Failed to retrieve storage size from chunk %ld\n", i);

        size_t decompressed_size = chunk_num_elements * dset_element_size;

        if (i == nchunks-1)
            decompressed_size = (dataset_num_elements - chunk_num_elements * i) * dset_element_size;

        // TODO: push 'offset' rather than 'mem_offset'
        data_blocks.push_back(std::make_tuple(addr, mem_offset, compressed_size, decompressed_size));
        mem_offset += decompressed_size;
    }

    return true;
}

bool DirectDataset::readChunked(hid_t dset_id, DirectFile *directfile, DeviceMemory &mm)
{
    std::vector<std::tuple<haddr_t, haddr_t, hsize_t, hsize_t>> data_blocks;
    std::vector<hsize_t> dims, cdims;
    bool retval = true;

    // Get information about each data chunk
    auto fspace_id = H5Dget_space(dset_id);
    if (parseChunks(dset_id, fspace_id, dims, cdims, data_blocks) == false)
    {
        H5Sclose(fspace_id);
        return false;
    }
    H5Sclose(fspace_id);

    // Read the compressed dataset via GDS to 'compressed_buffer' and decompress to 'mm'
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    const int chunk_size = 1 << 16;
    nvcomp::SnappyManager nvcomp_manager{chunk_size, stream};

    Benchmark benchmark;
    DeviceMemory compressed_buffer(mm.size);

    for (size_t i=0, compressed_offset=0, decompressed_offset=0; i<data_blocks.size(); ++i)
    {
        auto file_offset = std::get<0>(data_blocks[i]);
        auto mem_offset = std::get<1>(data_blocks[i]);
        auto chunk_size = std::get<2>(data_blocks[i]);
        auto decompressed_size = std::get<3>(data_blocks[i]);

        auto comp_data = ((uint8_t *) compressed_buffer.dev_mem);
        auto decomp_data = ((uint8_t *) mm.dev_mem) + decompressed_offset;

        // Read to the given compressed buffer offset
        H5FD_ctl_readp_args_t readp_args = {
            .type = H5FD_MEM_DEFAULT,
            .fapl_id = H5P_DEFAULT,
            .addr = file_offset,
            .size = chunk_size,
            .output_buf_offset = compressed_offset
        };
        uint64_t opcode = H5FD_CTL__READP;
        uint64_t flags = H5FD_CTL__FAIL_IF_UNKNOWN_FLAG;

        if (H5FDctl(directfile->file_driver, opcode, flags, (void *) &readp_args, (void **) &comp_data) < 0)
        {
            fprintf(stderr, "Failed to read file data\n");
            return false;
        }

        // Decompress
        auto decomp_config = nvcomp_manager.configure_decompression(comp_data);
        nvcomp_manager.decompress(decomp_data, comp_data, decomp_config);

        compressed_offset += chunk_size;
        decompressed_offset += decompressed_size;
    }

    cudaStreamDestroy(stream);
    benchmark.print("Read/decompress loop");

    return retval;
}

bool DirectDataset::readContiguous(hid_t dset_id, DirectFile *directfile, DeviceMemory &mm)
{
    bool retval = true, need_unsetenv = false;

    // Define the I/O transfer size and thread count
    if (getenv("H5_GDS_VFD_IO_THREADS") == NULL)
    {
        int max_threads = omp_get_max_threads();
        float tmp = ((float) mm.size) / MB(1);
        if (tmp < max_threads)
            max_threads = ((int) tmp) + 1;

        std::stringstream ss;
        ss << max_threads;
        setenv("H5_GDS_VFD_IO_THREADS", ss.str().c_str(), 1);
        need_unsetenv = true;
    }

    auto offset = H5Dget_offset(dset_id);
    auto count = H5Dget_storage_size(dset_id);

    if (H5FDread(directfile->file_driver, H5FD_MEM_DEFAULT, H5P_DEFAULT, offset, count, mm.dev_mem) < 0)
    {
        fprintf(stderr, "Failed to read contiguous file data via H5FDread\n");
        retval = false;
    }

    if (need_unsetenv)
        unsetenv("H5_GDS_VFD_IO_THREADS");

    return retval;
}

bool DirectDataset::copyToHost(DeviceMemory &mm, void **host_mem)
{
    cudaError_t err = cudaMemcpy(*host_mem, mm.dev_mem, mm.size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        const char *msg = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error: %s\n", msg);
    }
    return err == cudaSuccess;
}
