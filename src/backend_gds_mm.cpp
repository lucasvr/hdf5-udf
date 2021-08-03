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
#include <string>
#include "backend_gds.h"

#define cudaCheckError() do { \
 cudaError_t e = cudaGetLastError(); \
 if (e!=cudaSuccess) { \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
   exit(0); \
 } \
} while(0)

// DeviceMemory: allocate and register memory on the device for GPUDirect I/O
DeviceMemory::DeviceMemory(size_t alloc_size) : dev_mem(NULL), size(alloc_size)
{
    cudaError_t err = cudaMalloc(&dev_mem, size);
    if (err != cudaSuccess)
    {
        const char *errmsg = cudaGetErrorString(err);
        fprintf(stderr, "Failed to allocate %jd page-locked bytes on GPU memory: %s\n",
                size, errmsg);
        dev_mem = NULL;
        return;
    }

    CUfileError_t gds_err = cuFileBufRegister(dev_mem, size, 0);
    if (gds_err.err != CU_FILE_SUCCESS)
    {
        fprintf(stderr, "Failed to register buffer: %s\n", CUFILE_ERRSTR(gds_err.err));
        cudaFree(dev_mem);
        dev_mem = NULL;
    }
}

DeviceMemory::~DeviceMemory()
{
    if (dev_mem)
    {
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

// DirectStorage: simple interface to register and deregister the GDS driver.
DirectStorage::DirectStorage()
{
    CUfileError_t gds_err = cuFileDriverOpen();
    if (gds_err.err != CU_FILE_SUCCESS)
        fprintf(stderr, "Failed to open the GDS driver: %s\n", CUFILE_ERRSTR(gds_err.err));
}

DirectStorage::~DirectStorage()
{
    CUfileError_t gds_err = cuFileDriverClose();
    if (gds_err.err != CU_FILE_SUCCESS)
        fprintf(stderr, "Failed to close the GDS driver: %s\n", CUFILE_ERRSTR(gds_err.err));
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
        fprintf(stderr, "Failed to enable the HDF5 direct I/O driver\n");
        H5Pclose(fapl_id);
        return false;
    }

    file_id = H5Fopen(hdf5_file.c_str(), H5F_ACC_RDONLY, fapl_id);
    if (file_id < 0)
    {
        fprintf(stderr, "Failed to open %s in Direct I/O mode\n", hdf5_file.c_str());
        H5Pclose(fapl_id);
        return false;
    }
    H5Pclose(fapl_id);

    // Get a *reference* to the file descriptor
    void *file_handle = NULL;
    herr_t err = H5Fget_vfd_handle(file_id, H5P_DEFAULT, (void **) &file_handle);
    if (err < 0)
    {
        fprintf(stderr, "Failed to get HDF5 file VFD handle\n");
        H5Fclose(file_id);
        file_id = -1;
        return false;
    }

    file_fd = *((int *) file_handle);

    // Configure Direct I/O
    memset(&gds_descr, 0, sizeof(gds_descr));
    gds_descr.handle.fd = file_fd;
    gds_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileError_t gds_err = cuFileHandleRegister(&gds_handle, &gds_descr);
    if (gds_err.err != CU_FILE_SUCCESS)
    {
        fprintf(stderr, "Failed to register file: %s\n", CUFILE_ERRSTR(gds_err.err));
        H5Fclose(file_id);
        file_id = -1;
        return false;
    }

    return true;
}

void DirectFile::close()
{
    if (file_id >= 0)
    {
        cuFileHandleDeregister(gds_handle);
        H5Fclose(file_id);
        file_fd = -1;
        file_id = -1;
    }
}

// DirectDataset: routines to read a dataset using GPUDirect Storage I/O
bool DirectDataset::read(hid_t dset_id, const CUfileHandle_t &gds_handle, DeviceMemory &mm)
{
    // Get dataset offset. This call only succeeds if the requested dataset
    // has contiguous storage.
    haddr_t dset_offset = H5Dget_offset(dset_id);
    if (dset_offset == HADDR_UNDEF)
    {
        fprintf(stderr, "Failed to get offset of HDF5 dataset\n");
        return false;
    }

    cudaSetDevice(0);
    cudaCheckError();

    // Read the dataset in one shot
    int n = cuFileRead(gds_handle, mm.dev_mem, mm.size, dset_offset, 0);
    if (n < 0)
    {
        fprintf(stderr, "Failed to read file data: %d\n", n);
        return false;
    }
    return true;
}

bool DirectDataset::copyToHost(DeviceMemory &mm, void **host_mem)
{
    cudaMemcpy(*host_mem, mm.dev_mem, mm.size, cudaMemcpyDeviceToHost);
    return true;
}