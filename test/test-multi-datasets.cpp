// hdf5-udf add_datasets.h5 test-multi-datasets.cpp
extern "C" void dynamic_dataset()
{
    auto ds1_data = lib.getData<int>("Dataset1");
    auto ds2_data = lib.getData<int>("Dataset2");
    auto udf_data = lib.getData<int>("VirtualDataset.cpp");
    auto udf_dims = lib.getDims("VirtualDataset.cpp");

    for (size_t i=0; i<udf_dims[0] * udf_dims[1]; ++i)
    {
        udf_data[i] = ds1_data[i] + ds2_data[i];
    }
}
