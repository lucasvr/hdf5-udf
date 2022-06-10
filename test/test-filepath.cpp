// hdf5-udf <file.h5> test-filepath.cpp Temperature.cpp:1000:double
#include <cstring>

extern "C" void dynamic_dataset()
{
    auto path = strrchr(lib.getFilePath(), '/');
	printf("HDF5 file path is '%s'\n", &path[1]);
}
