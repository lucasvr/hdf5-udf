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

static const std::vector<DatasetTypeInfo> numerical_types = {
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
static const std::vector<DatasetTypeInfo> misc_types = {
    {"compound",  "void*", H5T_COMPOUND,      -1},
    {"string",    "char*", H5T_C_S1,          -1},
};

// Helper functions
static bool sameDatatype(hid_t a, hid_t b)
{
    if (a == -1 || b == -1)
        return false;
    else if (a == H5T_COMPOUND)
        return H5Tget_class(b) == H5T_COMPOUND;
    else if (a == H5T_C_S1)
        return H5Tget_class(b) == H5T_STRING;
    return H5Tequal(a, b);
}

static bool isNativeDatatype(hid_t hdf5_datatype)
{
    if (hdf5_datatype == H5T_COMPOUND ||
        hdf5_datatype == H5T_STRING ||
        H5Tget_class(hdf5_datatype) == H5T_COMPOUND ||
        H5Tget_class(hdf5_datatype) == H5T_STRING)
        return false;
    return true;
}

const char *getDatatypeName(hid_t hdf5_datatype)
{
    if (hdf5_datatype != -1)
    {
        for (auto &info: numerical_types)
            if (sameDatatype(info.hdf5_datatype_id, hdf5_datatype))
                return info.datatype.c_str();
        for (auto &info: misc_types)
            if (sameDatatype(info.hdf5_datatype_id, hdf5_datatype))
                return info.datatype.c_str();
    }
    return NULL;
}

const char *getCastDatatype(hid_t hdf5_datatype)
{
    if (hdf5_datatype != -1)
    {
        for (auto &info: numerical_types)
            if (sameDatatype(info.hdf5_datatype_id, hdf5_datatype))
                return info.declaration.c_str();
        for (auto &info: misc_types)
            if (sameDatatype(info.hdf5_datatype_id, hdf5_datatype))
                return info.declaration.c_str();
    }
    return NULL;
}

size_t getHdf5Datatype(std::string datatype)
{
    if (datatype.size())
    {
        for (auto &info: numerical_types)
            if (info.datatype.compare(datatype) == 0)
                return info.hdf5_datatype_id;
        for (auto &info: misc_types)
            if (info.datatype.compare(datatype) == 0)
                return info.hdf5_datatype_id;
    }
    return -1;
}

hid_t getStorageSize(hid_t hdf5_datatype)
{
    if (hdf5_datatype != -1)
    {
        for (auto &info: numerical_types)
            if (sameDatatype(info.hdf5_datatype_id, hdf5_datatype))
                return info.datatype_size;
        for (auto &info: misc_types)
            if (sameDatatype(info.hdf5_datatype_id, hdf5_datatype))
            {
                return isNativeDatatype(info.hdf5_datatype_id) ?
                    info.datatype_size :
                    H5Tget_size(hdf5_datatype);
            }
    }
    return -1;
}

std::vector<CompoundMember> getCompoundMembers(hid_t hdf5_datatype)
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
        auto datatype_name = ::getDatatypeName(type);
        if (datatype_name == NULL)
        {
            fprintf(stderr, "Unsupported HDF5 datatype %#lx of compound member '%s'\n",
                type, name);
            H5free_memory(name);
            return std::vector<CompoundMember>();
        }

        // Take the pointer token ('*') off the cast data type string iff
        // the datatype is not a variable-sized string or a compound.
        std::string decl_datatype = ::getCastDatatype(type) ? : "";
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

// DatasetInfo class implementation
DatasetInfo::DatasetInfo(
    std::string in_name,
    std::vector<hsize_t> in_dims,
    std::string in_datatype,
    hid_t in_hdf5_datatype)
    :
    name(in_name),
    datatype(in_datatype),
    hdf5_datatype(in_hdf5_datatype),
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

    reopenDatatype();
}

DatasetInfo::DatasetInfo(const DatasetInfo &other)
{
    *this = other;
    reopenDatatype();
}

DatasetInfo::~DatasetInfo()
{
    if (hdf5_datatype != -1)
        H5Tclose(hdf5_datatype);
}

void DatasetInfo::reopenDatatype()
{
    if (hdf5_datatype != -1)
        hdf5_datatype = H5Tcopy(hdf5_datatype);
}

size_t DatasetInfo::getGridSize() const
{
    return std::accumulate(
        std::begin(dimensions),
        std::end(dimensions),
        1, std::multiplies<hsize_t>());
}

const char *DatasetInfo::getDatatypeName() const
{
    return ::getDatatypeName(hdf5_datatype);
}

size_t DatasetInfo::getHdf5Datatype() const
{
    return ::getHdf5Datatype(datatype);
}

hid_t DatasetInfo::getStorageSize() const
{
    return ::getStorageSize(hdf5_datatype);
}

const char *DatasetInfo::getCastDatatype() const
{
    return ::getCastDatatype(hdf5_datatype);
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
    return ::getCompoundMembers(hdf5_datatype);
}

CompoundMember DatasetInfo::getStringDeclaration(bool is_varstring, size_t size) const
{
    CompoundMember member = {
        .name = name,
        .type = is_varstring ? "char*" : "char",
        .offset = 0,
        .size = size,
        .is_char_array = is_varstring == false,
    };
    return member;
}