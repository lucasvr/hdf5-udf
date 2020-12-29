// hdf5-udf <file.h5> test-string.cpp Temperature.cpp:1000:double
extern "C" void dynamic_dataset()
{
    auto input_string = lib.getData<dataset1_t>("Dataset1");
    auto input_dims = lib.getDims("Dataset1");
    auto udf_data = lib.getData<double>("Temperature.cpp");
    auto udf_dims = lib.getDims("Temperature.cpp");

    for (size_t i=0; i<input_dims[0]; ++i) {
        printf("%s\n", input_string[i].value);
    }

    for (size_t i=0; i<udf_dims[0]; ++i)
        udf_data[i] = i * 1.0;
}
