/*
 * Simple example: combines data from two existing datasets using
 * CUDA and GPUDirect Storage I/O.
 *
 * To embed it in an existing HDF5 file, run:
 * $ hdf5-udf example-add_datasets.h5 example-add_datasets.cu
 *
 */

__global__ void add(int *a, int *b, int *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = a[i] + b[i];
}

extern "C" void dynamic_dataset()
{
    auto ds1_data = lib.getData<int>("Dataset1");
    auto ds2_data = lib.getData<int>("Dataset2");
    auto udf_data = lib.getData<int>("UserDefinedDataset");
    auto udf_dims = lib.getDims("UserDefinedDataset");

    auto size = udf_dims[0] * udf_dims[1];
    add<<<(size+255)/256, 256>>>(ds1_data, ds2_data, udf_data);
}
