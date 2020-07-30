/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: dataset.cpp
 *
 * High-level interfaces for information retrieval from HDF5 datasets.
 */

#include "dataset.h"

static std::vector<DatasetTypeInfo> dataset_type_info = {
    {"int16",  "int16_t*",  H5T_STD_I16LE,  sizeof(int16_t)},
    {"int32",  "int32_t*",  H5T_STD_I32LE,  sizeof(int32_t)},
    {"int64",  "int64_t*",  H5T_STD_I64LE,  sizeof(int64_t)},
    {"uint16", "uint16_t*", H5T_STD_U16LE,  sizeof(uint16_t)},
    {"uint32", "uint32_t*", H5T_STD_U32LE,  sizeof(uint32_t)},
    {"uint64", "uint64_t*", H5T_STD_U64LE,  sizeof(uint64_t)},
    {"float",  "float*",    H5T_IEEE_F32LE, sizeof(float)},
    {"double", "double*",   H5T_IEEE_F64LE, sizeof(double)},
};

DatasetInfo::DatasetInfo() :
    name(""),
    datatype(""),
    hdf5_datatype(-1),
    data(NULL)
{
}

DatasetInfo::DatasetInfo(std::string in_name, std::vector<hsize_t> in_dims, std::string in_datatype) :
    name(in_name),
    datatype(in_datatype),
    hdf5_datatype(-1),
    dimensions(in_dims),
    data(NULL)
{
    std::stringstream ss;
    for (size_t i=0; i<dimensions.size(); ++i) {
        ss << dimensions[i];
        if (i < dimensions.size()-1)
            ss << "x";
    }
    dimensions_str = ss.str();
}

size_t DatasetInfo::getGridSize()
{
    return std::accumulate(
        std::begin(dimensions),
        std::end(dimensions),
        1, std::multiplies<hsize_t>());
}

const char *DatasetInfo::getDatatype()
{
    for (auto &info: dataset_type_info)
        if (H5Tequal(info.hdf5_datatype_id, hdf5_datatype))
            return info.datatype.c_str();
    return NULL;
}

size_t DatasetInfo::getHdf5Datatype()
{
    for (auto &info: dataset_type_info)
        if (info.datatype.compare(datatype) == 0)
            return info.hdf5_datatype_id;
    return -1;
}

hid_t DatasetInfo::getStorageSize()
{
    for (auto &info: dataset_type_info)
        if (info.datatype.compare(datatype) == 0)
            return info.datatype_size;
    return -1;
}

const char *DatasetInfo::getCastDatatype()
{
    for (auto &info: dataset_type_info)
        if (info.datatype.compare(datatype) == 0)
            return info.declaration.c_str();
    return NULL;
}

void DatasetInfo::printInfo(std::string dataset_type)
{
    printf("%s dataset: %s, resolution=%lld",
        dataset_type.c_str(), name.c_str(), dimensions[0]);
    for (size_t i=1; i<dimensions.size(); ++i)
        printf("x%lld", dimensions[i]);
    printf(", datatype=%s\n", datatype.c_str());
}