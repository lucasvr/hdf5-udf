// hdf5-udf <file.h5> test-compound-string.cpp Temperature.cpp:1000:double
extern "C" void dynamic_dataset()
{
    auto compound     = lib.getData<dataset1_t>("Dataset1");
    auto strings      = lib.getData<dataset2_t>("Dataset2");
    auto strings_dims = lib.getDims("Dataset2");
    auto udf_data     = lib.getData<double>("Temperature.cpp");
    auto udf_dims     = lib.getDims("Temperature.cpp");

    int string_index = 0;
    for (size_t i=0; i<udf_dims[0]; ++i)
    {
        printf("serial: %d, location: %s, temperature: %f, pressure: %f, string: %s\n",
            compound[i].serial_number,
            compound[i].location,
            compound[i].temperature,
            compound[i].pressure,
            lib.string(strings[string_index++]));
        if (string_index == strings_dims[0])
            string_index = 0;
        udf_data[i] = compound[i].temperature;
    }
}
