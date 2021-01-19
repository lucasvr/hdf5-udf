/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: dataset.h
 *
 * High-level interfaces for information retrieval from HDF5 datasets.
 */
#ifndef __dataset_h
#define __dataset_h

#include <hdf5.h>
#include <vector>
#include <string>
#include <numeric>
#include <sstream>

/* Default size of H5T_C_S1 strings in UDFs */
#define DEFAULT_UDF_STRING_SIZE 32

struct CompoundMember {
    std::string name;
    std::string type;
    size_t offset;
    size_t size;
    bool is_char_array;
};

/* Dataset information */
class DatasetInfo {
public:
    DatasetInfo(
        std::string in_name, 
        std::vector<hsize_t> in_dims,
        std::string in_datatype,
        hid_t in_hdf5_datatype);
    DatasetInfo(const DatasetInfo &other);
    ~DatasetInfo();

    void reopenDatatype();
    size_t getGridSize() const;
    const char *getDatatypeName() const;
    size_t getHdf5Datatype() const;
    hid_t getStorageSize() const;
    const char *getCastDatatype() const;
    void printInfo(std::string dataset_type) const;
    std::vector<CompoundMember> getCompoundMembers() const;
    CompoundMember getStringDeclaration(bool is_varstring, size_t size) const;

    std::string name;                /* Dataset name */
    std::string datatype;            /* Datatype, given as string */
    std::string dimensions_str;      /* Dimensions, given as string */
    hid_t hdf5_datatype;             /* Datatype, given as HDF5 type */
    std::vector<hsize_t> dimensions; /* Dataset dimensions */
    std::vector<CompoundMember> members; /* Compound member names and types */
    void *data;                      /* Allocated buffer to hold dataset data */
};

/* Helper functions */
const char *getDatatypeName(hid_t hdf5_datatype);
const char *getCastDatatype(hid_t hdf5_datatype);
size_t getHdf5Datatype(std::string datatype);
hid_t getStorageSize(hid_t hdf5_datatype);
std::vector<CompoundMember> getCompoundMembers(hid_t hdf5_datatype);


#endif /* __dataset_h */
