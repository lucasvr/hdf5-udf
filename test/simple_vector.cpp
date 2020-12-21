// hdf5-udf simple_vector.h5 simple_vector.cpp Simple.cpp:1500:float
extern "C" void dynamic_dataset()
{
    auto udf_data = lib.getData<float>("Simple.cpp");
    auto udf_dims = lib.getDims("Simple.cpp");
    for (size_t i=0; i<udf_dims[0]; ++i)
        udf_data[i] = i;
}