/*
 * Simple example: combines data from two existing datasets using
 * CUDA and GPUDirect Storage I/O.
 *
 * To embed it in an existing HDF5 file, run:
 * $ hdf5-udf example-add_datasets.h5 example-add_datasets.cu
 *
 */
#include <math.h>

__global__ void add(int *a, int *b, int *out, size_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] + b[i];
}

extern "C" void dynamic_dataset()
{
    auto ds1_data = lib.getData<int>("Dataset1");
    auto ds2_data = lib.getData<int>("Dataset2");
    auto udf_data = lib.getData<int>("UserDefinedDataset");
    auto udf_dims = lib.getDims("UserDefinedDataset");

    size_t n = udf_dims[0] * udf_dims[1];
    int block_size = 1024;
    int grid_size = (int) ceil((float) (n * sizeof(int))/block_size);
    add<<<grid_size, block_size>>>(ds1_data, ds2_data, udf_data, n);
}
