/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: createh5.cpp
 *
 * Creates HDF5 files that can be used for testing purposes.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <string>
#include <sstream>
#include <fstream>
#include <functional>
#include <map>

// https://github.com/lucasvr/snappy-cuda
#ifndef SNAPPY_CUDA_FILTER_ID
#define SNAPPY_CUDA_FILTER_ID   32003
#endif

// https://github.com/lucasvr/nvcomp-iofilter
#ifndef NVCOMP_FILTER_ID
#define NVCOMP_FILTER_ID        32004
#endif

hsize_t NATIVE_DIM0 = 100;
hsize_t NATIVE_DIM1 = 50;

hsize_t CHUNK_DIM0 = NATIVE_DIM0;
hsize_t CHUNK_DIM1 = 25;

const std::map<std::string, H5Z_filter_t> filter_map {
    {"deflate", H5Z_FILTER_DEFLATE},
    {"snappy-cuda", SNAPPY_CUDA_FILTER_ID},
    {"nvcomp", NVCOMP_FILTER_ID},
};

int create_regular_dataset(hid_t file_id, std::string compression, int count);
int create_compound_dataset_nostring(hid_t file_id, bool simple_layout, std::string compression, int count);
int create_compound_dataset_string(hid_t file_id, bool simple_layout, std::string compression, int count);
int create_compound_dataset_varstring(hid_t file_id, bool simple_layout, std::string compression, int count);
int create_native_int32(hid_t file_id, std::string compression, int count);
int create_native_string(hid_t file_id, std::string compression, int count);
int create_native_varstring(hid_t file_id, std::string compression, int count);

std::string getOptionValue(int argc, char **argv, const char *option, const char *default_value)
{
    for (int i=1; i<argc; ++i)
    {
        char *start = strstr(argv[i], option);
        if (start == argv[i] && strlen(argv[i]) == strlen(option))
        {
            // No-argument option (e.g., --help). Return "1"
            return "1";
        }
        else if (start == argv[i])
        {
            char *value = &start[strlen(option) + strlen("=")];
            return std::string(value);
        }
    }
    return std::string(default_value);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::stringstream ss;
        for (auto const &entry: filter_map)
            ss << entry.first << ", ";
        auto supported_filters = ss.str().substr(0, ss.str().length()-2);

        fprintf(stdout, "Syntax: %s  <options>\n", argv[0]);
        fprintf(stdout, "Available options are:\n"
            "  --compound=TYPE    Create a compound dataset with a predefined structure\n"
            "                     Valid options for TYPE are:\n"
            "                     NOSTRING_SIMPLE    (simple layout, no string members)\n"
            "                     NOSTRING_MIXED     (mixed layout, no string members)\n"
            "                     STRING_SIMPLE      (simple layout, including a fixed-sized string member)\n"
            "                     STRING_MIXED       (mixed layout, including a fixed-sized string member)\n"
            "                     VARSTRING_SIMPLE   (simple layout, including a variable-sized string member)\n"
            "                     VARSTRING_MIXED    (mixed layout, including a variable-sized string member)\n"
            "  --datatype=TYPE    Create a dataset with a predefined native type (default: INT32)\n"
            "                     Valid options for TYPE are:\n"
            "                     INT32              (an integer-based dataset)\n"
            "                     STRING             (a fixed-sized string dataset)\n"
            "                     VARSTRING          (a variable-sized string dataset)\n"
            "  --count=N          Create this many datasets in the output file (default: 0)\n"
            "  --dims=X,Y         X and Y dimensions (default: %lu,%lu)\n"
            "  --cdims=X,Y        Chunked X and Y dimensions (default: %lu,%lu)\n"
            "  --compress=FILTER  Compress the dataset(s) with one of the supported filters:\n"
            "                     %s\n"
            "  --out=FILE         Output file name (truncates FILE if it already exists)\n\n",
            NATIVE_DIM0, NATIVE_DIM1, CHUNK_DIM0, CHUNK_DIM1, supported_filters.c_str());
        return 1;
    }

    // Argument processing
    std::stringstream default_dims, default_cdims;
    default_dims << NATIVE_DIM0 << "," << NATIVE_DIM1;
    default_cdims << CHUNK_DIM0 << "," << CHUNK_DIM1;

    std::string compound = getOptionValue(argc, argv, "--compound", "");
    std::string datatype = getOptionValue(argc, argv, "--datatype", "INT32");
    int dataset_count = atoi(getOptionValue(argc, argv, "--count", "0").c_str());
    std::string compression = getOptionValue(argc, argv, "--compress", "");
    std::string hdf5_file = getOptionValue(argc, argv, "--out", "");
    if (hdf5_file.size() == 0)
    {
        fprintf(stderr, "Error: missing output file (--out=FILE)\n");
        return 1;
    }

    std::string dims = getOptionValue(argc, argv, "--dims", default_dims.str().c_str());
    sscanf(dims.c_str(), "%lu,%lu", &NATIVE_DIM0, &NATIVE_DIM1);

    std::string cdims = getOptionValue(argc, argv, "--cdims", default_cdims.str().c_str());
    sscanf(cdims.c_str(), "%lu,%lu", &CHUNK_DIM0, &CHUNK_DIM1);

    // Create (or truncate an existing) output file
    hid_t file_id = H5Fcreate(hdf5_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0)
    {
        fprintf(stderr, "Failed to create file %s\n", hdf5_file.c_str());
        return 1;
    }

    struct {
        const char *type;
        bool simple_layout;
        int (*func)(hid_t, bool, std::string, int);
    } compound_functions[] = {
        {"NOSTRING_SIMPLE",  true,  create_compound_dataset_nostring},
        {"NOSTRING_MIXED",   false, create_compound_dataset_nostring},
        {"STRING_SIMPLE",    true,  create_compound_dataset_string},
        {"STRING_MIXED",     false, create_compound_dataset_string},
        {"VARSTRING_SIMPLE", true,  create_compound_dataset_varstring},
        {"VARSTRING_MIXED",  false, create_compound_dataset_varstring},
        {NULL, false, NULL}
    };

    struct {
        const char *type;
        int (*func)(hid_t, std::string, int);
    } native_functions[] = {
        {"INT32",      create_native_int32},
        {"STRING",     create_native_string},
        {"VARSTRING",  create_native_varstring},
        {NULL, NULL}
    };

    int ret = 0, count = 1;
    while (true)
    {
        if (compound.size())
        {
            ret = -1;
            for (int i=0; compound_functions[i].type != NULL; ++i)
            {
                auto entry = &compound_functions[i];
                if (compound.compare(entry->type) == 0)
                    ret = entry->func(file_id, entry->simple_layout, compression, count);
            }
            if (ret == -1)
            {
                fprintf(stderr, "Invalid compound type '%s' requested\n", compound.c_str());
                break;
            }
            count++;
        }
        if (datatype.size())
        {
            ret = -1;
            for (int i=0; native_functions[i].type != NULL; ++i)
            {
                auto entry = &native_functions[i];
                if (datatype.compare(entry->type) == 0)
                    ret = entry->func(file_id, compression, count);
            }
            if (ret == -1)
            {
                fprintf(stderr, "Invalid datatype '%s' requested\n", datatype.c_str());
                break;
            }
            count++;
        }
        if (ret != 0 || count > dataset_count)
            break;
    }

    H5Fclose(file_id);
    return ret;
}

// Files with native data types

hid_t __make_compression_plist(std::string compression)
{
    auto filter_info = filter_map.find(compression);
    if (filter_info == filter_map.end())
    {
        fprintf(stderr, "Requested compression filter not supported\n");
        return -1;
    }
    hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
    if (plist_id < 0)
    {
        fprintf(stderr, "Failed to create property list\n");
        return -1;
    }

    size_t cd_nvalues = 0;
    unsigned int cd_values[4];
    if (filter_info->second == H5Z_FILTER_DEFLATE)
    {
        // Set compression level
        cd_nvalues = 1;
        cd_values[0] = 9;
    }

    herr_t ret = H5Pset_filter(plist_id, filter_info->second, H5Z_FLAG_MANDATORY, cd_nvalues, cd_values);
    if (ret < 0)
    {
        fprintf(stderr, "Failed to configure compression filter. Please check that $HDF5_PLUGIN_PATH is set\n");
        H5Pclose(plist_id);
        return -1;
    }
    hsize_t cdims[2] = {CHUNK_DIM0, CHUNK_DIM1};
    ret = H5Pset_chunk(plist_id, 2, cdims);
    if (ret < 0)
    {
        fprintf(stderr, "Failed to set chunk size\n");
        H5Pclose(plist_id);
        return -1;
    }

    return plist_id;
}

int __create_native_dataset(
    hid_t file_id, hid_t space_id, hid_t type_id, hid_t mem_type, int count, void *data, std::string compression)
{
    int retval = 0;
    char name[64];
    snprintf(name, sizeof(name)-1, "Dataset%d", count);

    hid_t plist_id = H5P_DEFAULT;
    if (compression.size())
    {
        plist_id = __make_compression_plist(compression);
        if (plist_id < 0)
            return 1;
    }

    hid_t dset_id = H5Dcreate(file_id, name, type_id, space_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    if (dset_id < 0)
    {
        fprintf(stderr, "Failed to create dataset\n");
        retval = 1;
    }
    else
    {
        herr_t ret = H5Dwrite(dset_id, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        if (ret < 0)
        {
            fprintf(stderr, "Error writing data to file\n");
            retval = 1;
        }
        H5Dclose(dset_id);
    }

    if (plist_id != H5P_DEFAULT)
        H5Pclose(plist_id);
    return retval;
}

int create_native_int32(hid_t file_id, std::string compression, int count)
{
    int32_t *data = new int[NATIVE_DIM0 * NATIVE_DIM1];
    for (hsize_t i=0; i<NATIVE_DIM0; ++i)
        for (hsize_t j=0; j<NATIVE_DIM1; ++j)
            data[i * NATIVE_DIM1 + j] = count * 10 * i + j;

    hsize_t dims[2] = {NATIVE_DIM0, NATIVE_DIM1};
    hid_t space_id = H5Screate_simple(2, dims, NULL);
    if (space_id < 0)
    {
        fprintf(stderr, "Failed to create dataspace\n");
        delete[] data;
        return 1;
    }

    int ret = __create_native_dataset(
        file_id, space_id, H5T_STD_I32LE, H5T_NATIVE_INT, count, (void *) data, compression);

    H5Sclose(space_id);
    delete[] data;
    return ret;
}

int __populate_string_data(
    int dims,
    int &current_dim,
    std::function<void(int, const char *)> write_fn)
{
    // We use the LICENSE file as input to string-based datasets
    std::ifstream textfile("../LICENSE");
    if (! textfile.is_open())
    {
        fprintf(stderr, "Failed to open '../LICENSE'\n");
        return 1;
    }
    std::string inputFileBuffer(
		(std::istreambuf_iterator<char>(textfile)),
        (std::istreambuf_iterator<char>()));

    std::string word;
    std::istringstream iss(inputFileBuffer);
    while (iss >> word && current_dim < dims)
    {
        write_fn(current_dim, word.c_str());
        current_dim++;
    }

    // Fill up the remaining dimensions with a static string, if needed
    while (current_dim < dims)
    {
        write_fn(current_dim, "<dummy>");
        current_dim++;
    }

    return 0;
}

int create_native_varstring(hid_t file_id, std::string compression, int count)
{
    // Prepare output data
    int current_dim = 0;
    char *data[NATIVE_DIM0];

    auto _callback = [&](int dim, const char *word)
    {
        data[dim] = strdup(word);
    };

    int ret = __populate_string_data(NATIVE_DIM0, current_dim, _callback);
    if (ret != 0)
        return 1;

    // Prepare HDF5 handles
    hsize_t dims[1] = {NATIVE_DIM0};
    hid_t space_id = H5Screate_simple(1, dims, NULL);
    if (space_id < 0)
    {
        fprintf(stderr, "Failed to create dataspace\n");
        return 1;
    }

    // Variable-length string type
    hid_t stringtype_id = H5Tcopy(H5T_C_S1);
    H5Tset_size(stringtype_id, H5T_VARIABLE);

    ret = __create_native_dataset(
        file_id, space_id, stringtype_id, stringtype_id, count, (void *) data, compression);
    
    H5Sclose(space_id);
    for (hsize_t dim=0; dim<NATIVE_DIM0; ++dim)
        free(data[dim]);
    return ret;
}

int create_native_string(hid_t file_id, std::string compression, int count)
{
    // Prepare output data
    const int len = 32;
    int current_dim = 0;
    char data[NATIVE_DIM0][len];
    memset(data, 0, sizeof(data));

    auto _callback = [&](int dim, const char *word)
    {
        snprintf(data[dim], len-1, "%s", word);
    };

    int ret = __populate_string_data(NATIVE_DIM0, current_dim, _callback);
    if (ret != 0)
        return 1;

    // Prepare HDF5 handles
    hsize_t dims[1] = {NATIVE_DIM0};
    hid_t space_id = H5Screate_simple(1, dims, NULL);
    if (space_id < 0)
    {
        fprintf(stderr, "Failed to create dataspace\n");
        return 1;
    }

    // Fixed-length string type
    hid_t stringtype_id = H5Tcopy(H5T_C_S1);
    H5Tset_size(stringtype_id, len);

    ret = __create_native_dataset(
        file_id, space_id, stringtype_id, stringtype_id, count, (void *) data[0], compression);
    
    H5Sclose(space_id);
    return ret;
}

// Files with compound data types
const int COMPOUND_DIM0 = 1000;

struct compound_varstring_t {
    int     serial_no;
    char   *location;
    double  temperature;
    double  pressure;
};

struct compound_string_t {
    int     serial_no;
    char    location[20];
    double  temperature;
    double  pressure;
};

struct compound_nostring_t {
    int     serial_no;
    double  temperature;
    double  pressure;
};

template <typename T>
int __create_compound_dataset(
    hid_t file_id,
    bool simple_layout,
    size_t varstr_size,
    int count,
    T data,
    std::string compression,
    std::function<void(hid_t, const char *, int)> string_fn)
{
    char name[64];
    snprintf(name, sizeof(name)-1, "Dataset%d", count);
    hsize_t dims[1] = {COMPOUND_DIM0};
    hid_t space_id = H5Screate_simple(1, dims, NULL);
    if (space_id < 0)
    {
        fprintf(stderr, "Failed to create dataspace\n");
        return 1;
    }

    hid_t plist_id = H5P_DEFAULT;
    if (compression.size())
    {
        plist_id = __make_compression_plist(compression);
        if (plist_id < 0)
        {
            H5Sclose(space_id);
            return 1;
        }
    }

    // The dataset layout is based on the example provided by the HDF Group at
    // https://support.hdfgroup.org/ftp/HDF5/examples/examples-by-api/hdf5-examples/1_8/C/H5T/h5ex_t_cmpd.c

    hid_t memtype_id = H5Tcreate(H5T_COMPOUND, sizeof(*data));
    H5Tinsert(memtype_id, "Serial number", HOFFSET(__typeof(*data), serial_no), H5T_NATIVE_INT);
    string_fn(memtype_id, "Location", -1);
    H5Tinsert(memtype_id, "Temperature (F)", HOFFSET(__typeof(*data), temperature), H5T_NATIVE_DOUBLE);
    H5Tinsert(memtype_id, "Pressure (inHg)", HOFFSET(__typeof(*data), pressure), H5T_NATIVE_DOUBLE);

    // If simple_layout==false, then we use a different disk layout to exercise HDF5-UDF's
    // ability to pad the compound structure. Note that the example at the URL above includes
    // a 'char *' to a string element that we don't include here at all times.

    hid_t filetype_id = H5Tcreate(H5T_COMPOUND, 8 + varstr_size + 8 + 8);
    H5Tinsert(filetype_id, "Serial number", 0, H5T_STD_I64LE);
    string_fn(filetype_id, "Location", 8);
    H5Tinsert(filetype_id, "Temperature (F)", 8 + varstr_size, H5T_IEEE_F64LE);
    H5Tinsert(filetype_id, "Pressure (inHg)", 8 + varstr_size + 8, H5T_IEEE_F64LE);

    // Create the compound dataset using either a simple approach (in which we write data
    // according to the memory layout) or a more complex one in which the memory and disk
    // layouts differ.
    int retval = 0;
    hid_t dset_id = H5Dcreate(file_id, name, simple_layout ? memtype_id : filetype_id, space_id,
        H5P_DEFAULT, plist_id, H5P_DEFAULT);
    if (dset_id < 0)
    {
        fprintf(stderr, "Failed to create dataset\n");
        retval = 1;
    }
    else
    {
        herr_t ret = H5Dwrite(dset_id, memtype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        if (ret < 0)
        {
            fprintf(stderr, "Error writing data to file\n");
            retval = 1;
        }
    }
    if (plist_id != H5P_DEFAULT)
        H5Pclose(plist_id);
    H5Tclose(filetype_id);
    H5Tclose(memtype_id);
    H5Dclose(dset_id);
    H5Sclose(space_id);
    return retval;
}

int create_compound_dataset_varstring(hid_t file_id, bool simple_layout, std::string compression, int count)
{
    compound_varstring_t data[COMPOUND_DIM0];
    memset(data, 0, sizeof(data));
    for (int i=0; i<COMPOUND_DIM0; ++i) {
        char loc[64];
        snprintf(loc, sizeof(loc)-1, "Location_%d", i);
        data[i].serial_no = i;
        // This is a short-lived program. Just let this one leak.
        data[i].location = strdup(loc);
        data[i].temperature = COMPOUND_DIM0/(i+1.0);
        data[i].pressure = COMPOUND_DIM0/((i+1)*2.0);
    }

    // Variable-length string type
    hid_t stringtype_id = H5Tcopy(H5T_C_S1);
    H5Tset_size(stringtype_id, H5T_VARIABLE);

    auto _callback = [&](hid_t type_id, const char *name, int offset)
    {
        H5Tinsert(
            type_id,
            name,
            offset != -1 ? offset : HOFFSET(compound_varstring_t, location),
            stringtype_id);
    };

    return __create_compound_dataset<compound_varstring_t *>(
        file_id, simple_layout, sizeof(hvl_t), count, data, compression, _callback);
}


int create_compound_dataset_string(hid_t file_id, bool simple_layout, std::string compression, int count)
{
    compound_string_t data[COMPOUND_DIM0];
    memset(data, 0, sizeof(data));
    for (int i=0; i<COMPOUND_DIM0; ++i) {
        data[i].serial_no = i;
        sprintf(data[i].location, "Location_%d", i);
        data[i].temperature = COMPOUND_DIM0/(i+1.0);
        data[i].pressure = COMPOUND_DIM0/((i+1)*2.0);
    }

    // Fixed-length string type
    hid_t stringtype_id = H5Tcopy(H5T_C_S1);
    size_t varstr_size = sizeof(data[0].location);
    H5Tset_size(stringtype_id, varstr_size);

    auto _callback = [&](hid_t type_id, const char *name, int offset)
    {
        H5Tinsert(
            type_id,
            name,
            offset != -1 ? offset : HOFFSET(compound_string_t, location),
            stringtype_id);
    };

    return __create_compound_dataset<compound_string_t *>(
        file_id, simple_layout, varstr_size, count, data, compression, _callback);
}

int create_compound_dataset_nostring(hid_t file_id, bool simple_layout, std::string compression, int count)
{
    compound_nostring_t data[COMPOUND_DIM0];
    memset(data, 0, sizeof(data));
    for (int i=0; i<COMPOUND_DIM0; ++i) {
        data[i].serial_no = i;
        data[i].temperature = COMPOUND_DIM0/(i+1.0);
        data[i].pressure = COMPOUND_DIM0/((i+1)*2.0);
    }

    auto _callback = [&](hid_t type_id, const char *name, int offset)
    {
    };

    // Provide a sizeof(hvl_t) to ensure the structure is padded when
    // simple_layout == false.
    return __create_compound_dataset<compound_nostring_t *>(
        file_id, simple_layout, sizeof(hvl_t), count, data, compression, _callback);
}
