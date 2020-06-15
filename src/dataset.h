/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: dataset.h
 *
 * High-level interfaces for information retrieval from HDF5 datasets.
 */
#ifndef __dataset_h
#define __dataset_h

#include <vector>
#include <string>
#include <numeric>
#include <sstream>

struct DatasetTypeInfo {
    DatasetTypeInfo(std::string dtype, std::string ddeclaration, hid_t did, hid_t dsize) :
        datatype(dtype),
        declaration(ddeclaration),
        hdf5_datatype_id(did),
        datatype_size(dsize) { }

    std::string datatype;
    std::string declaration;
    hid_t hdf5_datatype_id;
    hid_t datatype_size;
};

std::vector<DatasetTypeInfo> dataset_type_info = {
    {"int16",  "int16_t*",  H5T_STD_I16LE,  sizeof(int16_t)},
    {"int32",  "int32_t*",  H5T_STD_I32LE,  sizeof(int32_t)},
    {"int64",  "int64_t*",  H5T_STD_I64LE,  sizeof(int64_t)},
    {"uint16", "uint16_t*", H5T_STD_U16LE,  sizeof(uint16_t)},
    {"uint32", "uint32_t*", H5T_STD_U32LE,  sizeof(uint32_t)},
    {"uint64", "uint64_t*", H5T_STD_U64LE,  sizeof(uint64_t)},
    {"float",  "float*",    H5T_IEEE_F32LE, sizeof(float)},
    {"double", "double*",   H5T_IEEE_F64LE, sizeof(double)},
};

/* Dataset information */
class DatasetInfo {
public:
    DatasetInfo() :
        name(""),
        datatype(""),
        hdf5_datatype(-1),
        data(NULL) { }

    DatasetInfo(std::string in_name, std::vector<hsize_t> in_dims, std::string in_datatype) :
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

    size_t getGridSize() {
        return std::accumulate(
            std::begin(dimensions),
            std::end(dimensions),
            1, std::multiplies<hsize_t>());
    }

    const char *getDatatype() {
        for (auto &info: dataset_type_info)
            if (H5Tequal(info.hdf5_datatype_id, hdf5_datatype))
                return info.datatype.c_str();
        return NULL;
    }

    size_t getHdf5Datatype() {
        for (auto &info: dataset_type_info)
            if (info.datatype.compare(datatype) == 0)
                return info.hdf5_datatype_id;
        return -1;
    }

    hid_t getStorageSize() {
        for (auto &info: dataset_type_info)
            if (info.datatype.compare(datatype) == 0)
                return info.datatype_size;
        return -1;
    }

    const char *getCastDatatype() {
        for (auto &info: dataset_type_info)
            if (info.datatype.compare(datatype) == 0)
                return info.declaration.c_str();
        return NULL;
    }

    void printInfo(std::string dataset_type) {
        printf("%s dataset: %s, resolution=%lld",
            dataset_type.c_str(), name.c_str(), dimensions[0]);
        for (size_t i=1; i<dimensions.size(); ++i)
            printf("x%lld", dimensions[i]);
        printf(", datatype=%s\n", datatype.c_str());
    }

    std::string name;                /* Dataset name */
    std::string datatype;            /* Datatype, given as string */
    std::string dimensions_str;      /* Dimensions, given as string */
    hid_t hdf5_datatype;             /* Datatype, given as HDF5 type */
    std::vector<hsize_t> dimensions; /* Dataset dimensions */
    void *data;                      /* Allocated buffer to hold dataset data */
};

#endif /* __dataset_h */
