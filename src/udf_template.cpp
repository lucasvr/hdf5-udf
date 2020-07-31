//
// HDF5-UDF: User-Defined Functions for HDF5
//
// File: udf_template.cpp
//
// HDF5 filter callbacks and main interface with the C++ API.
//
#include <sys/types.h>
#include <string>
#include <vector>

// The following variables are populated by our HDF5 filter
std::vector<void *> hdf5_udf_data;
std::vector<const char *> hdf5_udf_names;
std::vector<const char *> hdf5_udf_types;
std::vector<std::vector<size_t>> hdf5_udf_dims;

// This is the API that user-defined-functions use to retrieve
// datasets they depend on.
class UserDefinedLibrary
{
public:
    template <class T>
    T *getData(std::string name);
    
    const char *getType(std::string name);
    
    std::vector<size_t> getDims(std::string name);
};

template <class T>
T *UserDefinedLibrary::getData(std::string name)
{
    for (size_t i=0; i<hdf5_udf_names.size(); ++i)
        if (name.compare(hdf5_udf_names[i]) == 0)
            return static_cast<T *>(hdf5_udf_data[i]);
    return NULL;
}

const char *UserDefinedLibrary::getType(std::string name)
{
    for (size_t i=0; i<hdf5_udf_names.size(); ++i)
        if (name.compare(hdf5_udf_names[i]) == 0)
            return hdf5_udf_types[i];
    return NULL;
}

std::vector<size_t> UserDefinedLibrary::getDims(std::string name)
{
    std::vector<size_t> dims;
    for (size_t i=0; i<hdf5_udf_names.size(); ++i)
        if (name.compare(hdf5_udf_names[i]) == 0)
            return hdf5_udf_dims[i];
    return dims;
}

UserDefinedLibrary lib;

// User-Defined Function

// user_callback_placeholder