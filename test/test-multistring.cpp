// hdf5-udf <file.h5> test-multistring.cpp Temperature.cpp:1000:double
#include <assert.h>

extern "C" void dynamic_dataset()
{
    auto input1_string = lib.getData<dataset1_t>("Dataset1");
    auto input1_dims = lib.getDims("Dataset1");
    auto input2_string = lib.getData<dataset2_t>("Dataset2");
    auto input2_dims = lib.getDims("Dataset2");
    auto udf_data = lib.getData<double>("Temperature.cpp");
    auto udf_dims = lib.getDims("Temperature.cpp");

    assert(input1_dims[0] == input2_dims[0]);
    for (size_t i=0; i<input1_dims[0]; ++i)
        printf("%s %s\n", lib.string(input1_string[i]), lib.string(input2_string[i]));
    
    for (size_t i=0; i<udf_dims[0]; ++i)
        udf_data[i] = i * 1.0;
}
