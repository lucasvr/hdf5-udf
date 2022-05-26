/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: backend_cuda_posix.cpp
 *
 * CUDA backend classes for I/O driven by the SEC2 VFD.
 */
#include <stdio.h>
#include <sys/types.h>
#include <string.h>
#include <unistd.h>
#include <string>
#include "backend_cuda.h"

// Methods from the DeviceMemory class
DeviceMemory::DeviceMemory(size_t alloc_size) :
    dev_mem(NULL),
    size(alloc_size),
    is_borrowed_mem(false)
{
    cudaError_t err = cudaMalloc(&dev_mem, size);
    if (err != cudaSuccess)
    {
        const char *errmsg = cudaGetErrorString(err);
        fprintf(stderr, "Failed to allocate %ld page-locked bytes on GPU memory: %s\n",
                size, errmsg);
        dev_mem = NULL;
    }
}

DeviceMemory::DeviceMemory(const DeviceMemory *src, size_t offset, size_t alloc_size) :
    dev_mem((void *)(((char *) src->dev_mem) + offset)),
    size(alloc_size),
    is_borrowed_mem(true)
{
}

DeviceMemory::~DeviceMemory()
{
    if (dev_mem && !is_borrowed_mem)
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

// Methods from the DmaFile class. The CUDA POSIX backend does not employ
// DMA-driven transfers.
DmaFile::DmaFile() : file_fd(-1), file_id(-1), fapl_id(-1)
{
}

DmaFile::~DmaFile()
{
    this->close();
}

bool DmaFile::open(std::string hdf5_file, bool warn_on_error=true)
{
    file_id = H5Fopen(hdf5_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0)
    {
        if (warn_on_error)
            FAIL("Failed to open %s\n", hdf5_file.c_str());
        return false;
    }
    return true;
}

void DmaFile::close()
{
    if (file_id >= 0)
    {
        H5Fclose(file_id);
        file_id = -1;
    }
}

// Methods from the DirectDataset class
bool DirectDataset::read(
    hid_t dset_id,
    hid_t hdf5_datatype,
    DmaFile *dmafile,
    DeviceMemory &mm)
{
    // DmaFile is not used by the POSIX VFD.
    (void) dmafile;

    char *host_mem = (char *) malloc(mm.size);
    if (! host_mem)
        FAIL("Failed to allocate host memory\n");

    bool read_ok = H5Dread(dset_id, hdf5_datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, host_mem) >= 0;
    if (read_ok)
    {
        if (! copyToDevice(host_mem, mm.size, mm))
        {
            free(host_mem);
            FAIL("Failed to copy dataset to GPU memory\n");
        }
    }

    free(host_mem);
    return read_ok;
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

bool DirectDataset::copyToDevice(const void *host_mem, hsize_t size, DeviceMemory &mm)
{
    if (mm.size < size)
    {
        fprintf(stderr, "copyToDevice: not enough space in device memory\n");
        return false;
    }

    cudaError_t err = cudaMemcpy(mm.dev_mem, host_mem, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        const char *msg = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error: %s\n", msg);
    }
    return err == cudaSuccess;
}