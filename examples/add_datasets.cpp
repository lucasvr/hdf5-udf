/*
 * Simple example: combines data from two existing datasets
 *
 * To embed it in an existing HDF5 file, run:
 * $ make files
 * $ hdf5-udf add_datasets.h5 add_datasets.c
 *
 * Note the absence of an output Dataset name in the call to
 * hdf5-udf: the tool determines it based on the calls to
 * lib.getData() made by this code. The resolution and
 * data types are determined to be the same as that of the
 * input datasets, Dataset1 and Dataset2.
 */
#include <stdio.h>

extern "C" void dynamic_dataset()
{
    auto ds1_data = lib.getData<int>("Dataset1");
    auto ds2_data = lib.getData<int>("Dataset2");
    auto udf_data = lib.getData<int>("VirtualDataset");
    auto udf_dims = lib.getDims("VirtualDataset");

    for (size_t i=0; i<udf_dims[0] * udf_dims[1]; ++i)
    {
        udf_data[i] = ds1_data[i] + ds2_data[i];
    }
}
