// hdf5-udf <file.h5> test-compound-nostring.cpp Temperature.cpp:1000:double
extern "C" void dynamic_dataset()
{
    auto compound = lib.getData<compound_dataset1>("Dataset1");
    auto udf_data = lib.getData<double>("Temperature.cpp");
    auto udf_dims = lib.getDims("Temperature.cpp");

    for (size_t i=0; i<udf_dims[0]; ++i)
    {
        printf("serial: %d, temperature: %f, pressure: %f\n",
            compound[i].serial_number,
            compound[i].temperature,
            compound[i].pressure);
        udf_data[i] = compound[i].temperature;
    }
}
