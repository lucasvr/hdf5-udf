/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: dataset.cpp
 *
 * High-level interfaces for information retrieval from HDF5 datasets.
 */

#include <string.h>
#include "dataset.h"

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

static std::vector<DatasetTypeInfo> numerical_types = {
    {"int8",   "int8_t*",   H5T_STD_I8LE,   sizeof(int8_t)},
    {"int16",  "int16_t*",  H5T_STD_I16LE,  sizeof(int16_t)},
    {"int32",  "int32_t*",  H5T_STD_I32LE,  sizeof(int32_t)},
    {"int64",  "int64_t*",  H5T_STD_I64LE,  sizeof(int64_t)},
    {"uint8",  "uint8_t*",  H5T_STD_U8LE,   sizeof(uint8_t)},
    {"uint16", "uint16_t*", H5T_STD_U16LE,  sizeof(uint16_t)},
    {"uint32", "uint32_t*", H5T_STD_U32LE,  sizeof(uint32_t)},
    {"uint64", "uint64_t*", H5T_STD_U64LE,  sizeof(uint64_t)},
    {"float",  "float*",    H5T_IEEE_F32LE, sizeof(float)},
    {"double", "double*",   H5T_IEEE_F64LE, sizeof(double)},
};

// misc_types are looked up by HDF5 *class* rather than *datatype*
static std::vector<DatasetTypeInfo> misc_types = {
    {"compound",  "void*", H5T_COMPOUND, -1},
    {"varstring", "char*", H5T_C_S1,     -1},
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

size_t DatasetInfo::getGridSize() const
{
    return std::accumulate(
        std::begin(dimensions),
        std::end(dimensions),
        1, std::multiplies<hsize_t>());
}

static bool sameDatatype(hid_t a, hid_t b)
{
    if (a == H5T_COMPOUND)
        return H5Tget_class(b) == H5T_COMPOUND;
    else if (a == H5T_C_S1)
        return H5Tget_class(b) == H5T_STRING;
    return H5Tequal(a, b);
}

const char *DatasetInfo::getDatatype() const
{
    if (hdf5_datatype != -1) {
        for (auto &info: numerical_types)
            if (H5Tequal(info.hdf5_datatype_id, hdf5_datatype))
                return info.datatype.c_str();
        for (auto &info: misc_types)
            if (sameDatatype(info.hdf5_datatype_id, hdf5_datatype))
                return info.datatype.c_str();
    }
    return NULL;
}

size_t DatasetInfo::getHdf5Datatype() const
{
    if (datatype.size()) {
        for (auto &info: numerical_types)
            if (info.datatype.compare(datatype) == 0)
                return info.hdf5_datatype_id;
        for (auto &info: misc_types)
            if (sameDatatype(info.hdf5_datatype_id, hdf5_datatype))
                return info.hdf5_datatype_id;
    }
    return -1;
}

hid_t DatasetInfo::getStorageSize() const
{
    if (datatype.size()) {
        for (auto &info: numerical_types)
            if (info.datatype.compare(datatype) == 0)
                return info.datatype_size;
        for (auto &info: misc_types)
            if (sameDatatype(info.hdf5_datatype_id, hdf5_datatype))
                return info.datatype_size;
    }
    return -1;
}

const char *DatasetInfo::getCastDatatype() const
{
    if (datatype.size()) {
        for (auto &info: numerical_types)
            if (info.datatype.compare(datatype) == 0)
                return info.declaration.c_str();
        for (auto &info: misc_types)
            if (sameDatatype(info.hdf5_datatype_id, hdf5_datatype))
                return info.declaration.c_str();
    }
    return NULL;
}

void DatasetInfo::printInfo(std::string dataset_type) const
{
    printf("%s dataset: %s, resolution=%lld",
        dataset_type.c_str(), name.c_str(), dimensions[0]);
    for (size_t i=1; i<dimensions.size(); ++i)
        printf("x%lld", dimensions[i]);
    size_t size = H5Tget_size(hdf5_datatype);
    printf(", datatype=%s, size=%ld\n", datatype.c_str(), size);
}

std::vector<CompoundMember> DatasetInfo::getCompoundMembers() const
{
    std::vector<CompoundMember> members;
    for (int i=0; i<H5Tget_nmembers(hdf5_datatype); ++i)
    {
        char *name = H5Tget_member_name(hdf5_datatype, i);
        hid_t type = H5Tget_member_type(hdf5_datatype, i);
        size_t off = H5Tget_member_offset(hdf5_datatype, i);
        hid_t hclass = H5Tget_member_class(hdf5_datatype, i);
        bool is_varstring = H5Tis_variable_str(type);
        size_t size = H5Tget_size(type);

        // Get this member's data type (e.g., 'int64', 'float') so we can lookup
        // its cast data type (e.g., 'int64_t*', 'float*') and storage size next.
        DatasetInfo member_info;
        member_info.hdf5_datatype = type;
        member_info.datatype = member_info.getDatatype() ? : "";
        if (member_info.datatype.size() == 0)
        {
            fprintf(stderr, "Unsupported HDF5 datatype %#lx of compound member '%s'\n",
                type, name);
            H5free_memory(name);
            return std::vector<CompoundMember>();
        }

        // Take the pointer token ('*') off the cast data type string iff
        // the datatype is not a variable-sized string or a compound.
        std::string decl_datatype = member_info.getCastDatatype() ? : "";
        auto ptr_token = decl_datatype.find("*");
        if (ptr_token != std::string::npos && ! is_varstring && hclass != H5T_COMPOUND)
            decl_datatype = decl_datatype.substr(0, ptr_token);

        // Push member information into the output vector
        CompoundMember member = {
            .name = name,
            .type = decl_datatype,
            .offset = off,
            .size = size,
            .is_char_array = hclass == H5T_STRING && is_varstring == false,
        };
        members.push_back(member);
        H5free_memory(name);
    }
    return members;
}

CompoundMember DatasetInfo::getStringDeclaration() const
{
    bool is_varstring = H5Tis_variable_str(hdf5_datatype);
    size_t size = H5Tget_size(hdf5_datatype);
    std::string decl_datatype = is_varstring ? "char*" : "char";

    CompoundMember member = {
        .name = name,
        .type = decl_datatype,
        .offset = 0,
        .size = size,
        .is_char_array = is_varstring == false,
    };
    return member;
}