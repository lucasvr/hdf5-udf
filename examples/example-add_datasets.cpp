/*
 * Simple example: combines data from two existing datasets
 *
 * To embed it in an existing HDF5 file, run:
 * $ make files
 * $ hdf5-udf example-add_datasets.h5 example-add_datasets.cpp
 *
 * Note the absence of an output Dataset name in the call to
 * hdf5-udf: the tool determines it based on the calls to
 * lib.getData() made by this code. The resolution and
 * data types are determined to be the same as that of the
 * input datasets, Dataset1 and Dataset2.
 */

#include <math.h>

extern "C" void dynamic_dataset()
{
    auto a = lib.getData<int>("Dataset1");
    auto b = lib.getData<int>("Dataset2");
    auto udf_data = lib.getData<int>("UserDefinedDataset-Cpp");
    auto udf_dims = lib.getDims("UserDefinedDataset-Cpp");

    // https://www.usgs.gov/core-science-systems/nli/landsat/landsat-modified-soil-adjusted-vegetation-index
    for (size_t i=0; i<udf_dims[0] * udf_dims[1]; ++i)
    {
        int n2 = (2 * a[i] + 1) * (2 * b[i] + 1);
        udf_data[i] = (2 * a[i] + 1 - sqrt((float) (n2 - 8 * (a[i] - b[i])))) / 2;
    }
}
