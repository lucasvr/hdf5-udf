/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: backend_gds_classes.cpp
 *
 * NVIDIA GPUDirect Storage backend helper classes.
 */
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <hdf5.h>
#include <H5FDdirect.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <omp.h>
#include <snappy-cuda/gds_interface.h>

#include <string>
#include <tuple>
#include <vector>
#include "backend_gds.h"
#include "dataset.h"
#include "os.h"

#define cudaCheckError() do { \
 cudaError_t e = cudaGetLastError(); \
 if (e!=cudaSuccess) { \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
   exit(0); \
 } \
} while(0)

#define ALIGN(_p, _width) (((unsigned int)_p + (_width-1)) & (0-_width))
#define ALIGN_LONG(_p, _width) (((long)_p + (_width-1)) & (0-_width))

#define FAIL(msg...) do { \
    fprintf(stderr, msg); \
    return false; \
} while (0)

// DeviceMemory: allocate and register memory on the device for GPUDirect I/O
DeviceMemory::DeviceMemory(size_t alloc_size, size_t aligned_alloc_size) :
    dev_mem(NULL),
    size(alloc_size),
    total_size(aligned_alloc_size ? aligned_alloc_size : alloc_size)
{
    cudaError_t err = cudaMalloc(&dev_mem, total_size);
    if (err != cudaSuccess)
    {
        const char *errmsg = cudaGetErrorString(err);
        fprintf(stderr, "Failed to allocate %ld page-locked bytes on GPU memory: %s\n",
                size, errmsg);
        dev_mem = NULL;
        return;
    }
}

DeviceMemory::~DeviceMemory()
{
    if (dev_mem)
    {
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

// DirectStorage: simple interface to register and deregister the GDS driver.
DirectStorage::DirectStorage() : is_opened(false)
{
}

void DirectStorage::open()
{
    CUfileError_t gds_err = cuFileDriverOpen();
    if (gds_err.err != CU_FILE_SUCCESS)
        fprintf(stderr, "Failed to open the GDS driver: %s\n", CUFILE_ERRSTR(gds_err.err));
    is_opened = gds_err.err == CU_FILE_SUCCESS;
}

DirectStorage::~DirectStorage()
{
    if (is_opened)
    {
        CUfileError_t gds_err = cuFileDriverClose();
        if (gds_err.err != CU_FILE_SUCCESS)
            fprintf(stderr, "Failed to close the GDS driver: %s\n", CUFILE_ERRSTR(gds_err.err));
    }
}

// DirectFile: open and configures a HDF5 file for Direct I/O
DirectFile::DirectFile() : file_fd(-1), file_id(-1)
{
}

DirectFile::~DirectFile()
{
    this->close();
}

bool DirectFile::open(std::string hdf5_file)
{
    // Open file with the Direct I/O driver
    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    if (fapl_id < 0)
    {
        fprintf(stderr, "Failed to create HDF5 file access property list\n");
        return false;
    }

    size_t alignment = 1024;
    size_t block_size = 0;
    size_t copy_buffer_size = 4096 * 8;
    if (H5Pset_fapl_direct(fapl_id, alignment, block_size, copy_buffer_size) < 0)
    {
        H5Pclose(fapl_id);
        FAIL("Failed to enable the HDF5 direct I/O driver\n");
    }

    std::string old_env = getenv("HDF5_USE_FILE_LOCKING") ? : "";
    os::setEnvironmentVariable("HDF5_USE_FILE_LOCKING", "FALSE");
    file_id = H5Fopen(hdf5_file.c_str(), H5F_ACC_RDONLY, fapl_id);
    os::setEnvironmentVariable("HDF5_USE_FILE_LOCKING", old_env);
    if (file_id < 0)
    {
        H5Pclose(fapl_id);
        FAIL("Failed to open %s in Direct I/O mode\n", hdf5_file.c_str());
    }
    H5Pclose(fapl_id);

    // Get a *reference* to the file descriptor
    void *file_handle = NULL;
    herr_t err = H5Fget_vfd_handle(file_id, H5P_DEFAULT, (void **) &file_handle);
    if (err < 0)
    {
        H5Fclose(file_id);
        file_id = -1;
        FAIL("Failed to get HDF5 file VFD handle\n");
    }

    file_fd = *((int *) file_handle);

    // Configure Direct I/O
    memset(&gds_descr, 0, sizeof(gds_descr));
    gds_descr.handle.fd = file_fd;
    gds_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileError_t gds_err = cuFileHandleRegister(&gds_handle, &gds_descr);
    if (gds_err.err != CU_FILE_SUCCESS)
    {
        H5Fclose(file_id);
        file_id = -1;
        file_fd = -1;
        FAIL("Failed to register file: %s\n", CUFILE_ERRSTR(gds_err.err));
    }

    return true;
}

void DirectFile::close()
{
    if (file_id >= 0)
    {
        cuFileHandleDeregister(gds_handle);
        H5Fclose(file_id);
        file_id = -1;
        file_fd = -1;
    }
}

// DirectDataset: routines to read a dataset using GPUDirect Storage I/O
bool DirectDataset::read(hid_t dset_id, DirectFile *directfile, DeviceMemory &mm)
{
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
    std::vector<std::tuple<haddr_t, haddr_t, hsize_t, hsize_t, DeviceMemory *>> &data_blocks)
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
            FAIL("Failed to retrieve information from chunk %lld\n", i);
        if (H5Dget_chunk_storage_size(dset_id, offset, &compressed_size) < 0)
            FAIL("Failed to retrieve storage size from chunk %lld\n", i);

        // Allocate a scratch memory area where the compressed chunk can be read to
        size_t total_size = ALIGN_LONG(compressed_size, 8) + 64;
        DeviceMemory *scratch_mm = new DeviceMemory(compressed_size, total_size);

        size_t decompressed_size = chunk_num_elements * dset_element_size;
        if (i == nchunks-1)
            decompressed_size = (dataset_num_elements - chunk_num_elements * i) * dset_element_size;

        // TODO: push 'offset' rather than 'mem_offset'
        data_blocks.push_back(std::make_tuple(addr, mem_offset, compressed_size, decompressed_size, scratch_mm));
        mem_offset += decompressed_size;
    }

    return true;
}

bool DirectDataset::readChunked(hid_t dset_id, DirectFile *directfile, DeviceMemory &mm)
{
    std::vector<std::tuple<haddr_t, haddr_t, hsize_t, hsize_t, DeviceMemory *>> data_blocks;
    std::vector<hsize_t> dims, cdims;
    bool retval = true;

    auto fspace_id = H5Dget_space(dset_id);
    if (parseChunks(dset_id, fspace_id, dims, cdims, data_blocks)  == false)
    {
        H5Sclose(fspace_id);
        return false;
    }
    H5Sclose(fspace_id);

    // TODO: this implementation only supports delegating workload to a single GPU
    cudaSetDevice(0);
    cudaCheckError();

    // Preallocate Snappy contexts for each OpenMP thread
    std::vector<void *> snappy_ctx;
    for (auto i=0; i<omp_get_max_threads(); ++i)
        snappy_ctx.push_back(decompressor_alloc());

    // Read the dataset
    #pragma omp parallel for
    for (size_t i=0; i<data_blocks.size(); ++i)
    {
        auto file_offset = std::get<0>(data_blocks[i]);
        auto mem_offset = std::get<1>(data_blocks[i]);
        auto chunk_size = std::get<2>(data_blocks[i]);
        auto decompressed_size = std::get<3>(data_blocks[i]);
        DeviceMemory *scratch_mm = std::get<4>(data_blocks[i]);

        // Transfer data from storage to GPU via DMA
        ssize_t n = cuFileRead(directfile->gds_handle, scratch_mm->dev_mem, chunk_size, file_offset, 0);
        if (n < 0)
        {
            // Cannot break from OpenMP structured block
            fprintf(stderr, "Failed to read file data: %jd (%s)\n", n, strerror(errno));
            delete scratch_mm;
            retval = false;
            continue;
        }
        else if (n != (ssize_t) chunk_size)
        {
            fprintf(stderr, "Requested to read %lld, received %jd\n", chunk_size, n);
        }

        // Decompress to output dataset memory buffer
        void *dst = (void *) (((char *) mm.dev_mem) + mem_offset);
        void *ctx = snappy_ctx[omp_get_thread_num()];
        decompressor_init(
            scratch_mm->dev_mem,
            scratch_mm->size,
            scratch_mm->total_size,
            dst,
            decompressed_size,
            ctx);
        decompressor_run(ctx);
        delete scratch_mm;
    }

    for (auto &ctx: snappy_ctx)
        decompressor_destroy(ctx);

    return retval;
}

bool DirectDataset::readContiguous(hid_t dset_id, DirectFile *directfile, DeviceMemory &mm)
{
    // Get dataset offset. This call only succeeds if the requested dataset
    // has contiguous storage.
    haddr_t dset_offset = H5Dget_offset(dset_id);
    if (dset_offset == HADDR_UNDEF)
    {
        fprintf(stderr, "Failed to get offset of HDF5 dataset\n");
        return false;
    }

    // TODO: this implementation only supports delegating workload to a single GPU
    cudaSetDevice(0);
    cudaCheckError();

    bool ret = true;

    // Read the dataset in one shot
    #pragma omp parallel for shared(ret)
    for (auto i=0; i<omp_get_num_threads(); ++i)
    {
        size_t my_offset = (mm.size / omp_get_num_threads() * i);
        size_t my_size = mm.size / omp_get_num_threads() * i;
        if (my_offset + my_size > mm.size)
            my_size = mm.size - mm.size;

        ssize_t n = cuFileRead(
            directfile->gds_handle,
            (void *) (((char *) mm.dev_mem) + my_offset),
            my_size,
            dset_offset + my_offset,
            0);
        if (n < 0)
        {
            fprintf(stderr, "Failed to read file data: %jd\n", n);
            #pragma omp critical
            ret = false;
        }
    }

    return ret;
}

bool DirectDataset::copyToHost(DeviceMemory &mm, void **host_mem)
{
    cudaMemcpy(*host_mem, mm.dev_mem, mm.size, cudaMemcpyDeviceToHost);
    return true;
}