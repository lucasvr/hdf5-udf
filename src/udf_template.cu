//
// HDF5-UDF: User-Defined Functions for HDF5
//
// File: udf_template.cpp
//
// HDF5 filter callbacks and main interface with the C++ API.
//
#include <sys/types.h>
#include <stdarg.h>
#include <math.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

// The following variables are populated by our HDF5 filter
std::vector<void *> hdf5_udf_data;
std::vector<const char *> hdf5_udf_names;
std::vector<const char *> hdf5_udf_types;
std::vector<std::vector<size_t> > hdf5_udf_dims;

// compound_declarations_placeholder

// This is the API that user-defined-functions use to retrieve
// datasets they depend on.
class UserDefinedLibrary
{
public:
    template <class T> T *getData(std::string name);
    const char *getType(std::string name);
    std::vector<size_t> getDims(std::string name);
    const char *string(const char *element);
    // methods_declaration_placeholder
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

const char *UserDefinedLibrary::string(const char *element)
{
    return element;
}

// API needed by the CUDA backend to workaround a bug in which the
// process that loaded the shared library cannot peek into the last
// CUDA error.
extern "C" int hdf5_udf_last_cuda_error()
{
    return static_cast<int>(cudaGetLastError());
}


// methods_implementation_placeholder

UserDefinedLibrary lib;

// User-Defined Function

// user_callback_placeholder
