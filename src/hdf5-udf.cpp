/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: hdf5-udf.cpp
 *
 * HDF5 filter callbacks and main interface with the Lua API.
 */
#include <dirent.h>
#include <H5PLextern.h>
#include <hdf5.h>
#include <stdlib.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <iostream>

#include "filter_id.h"
#include "dataset.h"
#include "debug.h"
#include "lua.hpp"
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

/* Lua context */
lua_State *State;
static lua_State *State

#define DATA_OFFSET(i)        (void *) (((char *) &State) + i)
#define NAME_OFFSET(i)        (void *) (((char *) &State) + 100 + i)
#define DIMS_OFFSET(i)        (void *) (((char *) &State) + 200 + i)
#define TYPE_OFFSET(i)        (void *) (((char *) &State) + 300 + i)
#define CAST_OFFSET(i)        (void *) (((char *) &State) + 400 + i)

extern "C" int index_of(const char *element)
{
    for (int index=0; index<100; ++index) {
        /* Set register key to get datasets name vector */
        lua_pushlightuserdata(State, NAME_OFFSET(index));
        lua_gettable(State, LUA_REGISTRYINDEX);
        const char *name = lua_tostring(State, -1);
        if (! strcmp(name, element))
            return index;
        else if (strlen(name) == 0)
            break;
    }
    fprintf(stderr, "Error: dataset %s not found\n", element);
    return -1;
}

/* Functions exported to the Lua library (udf.lua) */
extern "C" void *get_data(const char *element)
{
    int index = index_of(element);
    if (index >= 0)
    {
        /* Get datasets contents */
        lua_pushlightuserdata(State, DATA_OFFSET(index)); 
        lua_gettable(State, LUA_REGISTRYINDEX);
        return lua_touserdata(State, -1);
    }
    return NULL;
}

extern "C" const char *get_type(const char *element)
{
    int index = index_of(element);
    if (index >= 0)
    {
        /* Set register key to get dataset type */
        lua_pushlightuserdata(State, TYPE_OFFSET(index));
        lua_gettable(State, LUA_REGISTRYINDEX);
        return lua_tostring(State, -1);
    }
    return NULL;
}

extern "C" const char *get_cast(const char *element)
{
    int index = index_of(element);
    if (index >= 0)
    {
        /* Set register key to get dataset type */
        lua_pushlightuserdata(State, CAST_OFFSET(index));
        lua_gettable(State, LUA_REGISTRYINDEX);
        return lua_tostring(State, -1);
    }
    return NULL;
}

extern "C" const char *get_dims(const char *element)
{
    int index = index_of(element);
    if (index >= 0)
    {
        /* Set register key to get dataset size */
        lua_pushlightuserdata(State, DIMS_OFFSET(index));
        lua_gettable(State, LUA_REGISTRYINDEX);
        return lua_tostring(State, -1);
    }
    return NULL;
}

bool callLua(
    std::vector<DatasetInfo> &input_datasets, DatasetInfo &output_dataset,
    char *bytecode, long lSize, const char* dtype)
{
    lua_State *L = luaL_newstate();
    State = L;

    lua_pushcfunction(L, luaopen_os);
    lua_call(L,0,0);
    lua_pushcfunction(L, luaopen_base);
    lua_call(L,0,0);
    lua_pushcfunction(L, luaopen_math);
    lua_call(L,0,0);
    lua_pushcfunction(L, luaopen_string);
    lua_call(L,0,0);
    lua_pushcfunction(L, luaopen_ffi);
    lua_call(L,0,0);
    lua_pushcfunction(L, luaopen_jit);
    lua_call(L,0,0);
    lua_pushcfunction(L, luaopen_package);
    lua_call(L,0,0);
    lua_pushcfunction(L, luaopen_table);
    lua_call(L,0,0);

    DatasetInfo empty_entry;
    std::vector<DatasetInfo> dataset_info;
    dataset_info.push_back(output_dataset);
    dataset_info.insert(
        dataset_info.end(), input_datasets.begin(), input_datasets.end());
    dataset_info.push_back(empty_entry);

    /* Populate vector of dataset names, sizes, and types */
    for (size_t i=0; i<dataset_info.size(); ++i)
    {
        /* Grid */
        lua_pushlightuserdata(L, DATA_OFFSET(i));
        lua_pushlightuserdata(L, (void *) dataset_info[i].data);
        lua_settable(L, LUA_REGISTRYINDEX);

        /* Name */
        lua_pushlightuserdata(L, NAME_OFFSET(i));
        lua_pushstring(L, dataset_info[i].name.c_str());
        lua_settable(L, LUA_REGISTRYINDEX);

        /* Dimensions */
        lua_pushlightuserdata(L, DIMS_OFFSET(i));
        lua_pushstring(L, dataset_info[i].dimensions_str.c_str());
        lua_settable(L, LUA_REGISTRYINDEX);

        /* Type */
        lua_pushlightuserdata(L, TYPE_OFFSET(i));
        lua_pushstring(L, dataset_info[i].getDatatype());
        lua_settable(L, LUA_REGISTRYINDEX);

        /* Type, used for casting purposes */
        lua_pushlightuserdata(L, CAST_OFFSET(i));
        lua_pushstring(L, dataset_info[i].getCastDatatype());
        lua_settable(L, LUA_REGISTRYINDEX);
    }

    int retValue = luaL_loadbuffer(L, bytecode, lSize, "hdf5_udf_bytecode");
    if (retValue != 0)
    {
        fprintf(stderr, "luaL_loadbuffer failed: %s\n", lua_tostring(L, -1));
        lua_close(L);
        return false;
    }
    if (lua_pcall(L, 0, 0 , 0) != 0)
    {
        fprintf(stderr, "lua_pcall failed to load bytecode: %s\n", lua_tostring(L, -1));
        lua_close(L);
        return false;
    }
    lua_getglobal(L, "dynamic_dataset");
    if (lua_pcall(L, 0, 0, 0) != 0)
    {
        fprintf(stderr, "lua_pcall failed to invoke entry point: %s\n", lua_tostring(L, -1));
        lua_close(L);
        return false;
    }
    lua_close(L);

    return true;
}

/* Retrieve the HDF5 file handle associated with a given dataset name */
hid_t getDatasetHandle(std::string dataset, bool *handle_from_procfs)
{
    /*
     * Get a list of open files from /proc. This is a workaround for the
     * lack of an HDF5 Filter API to access the underlying file descriptor.
     */
    auto getProcCandidates = [&]()
    {
        const char *proc = "/proc/self/fd";
        std::vector<std::string> out;
        DIR *d = opendir(proc);
        if (!d)
            return out;

        struct dirent *e;
        while ((e = readdir(d)) != NULL)
        {
            struct stat s;
            auto fname = std::string(proc) + "/" + std::string(e->d_name);
            if (stat(fname.c_str(), &s) == 0 && S_ISREG(s.st_mode))
            {
                char target[PATH_MAX];
                memset(target, 0, sizeof(target));
                if (readlink(fname.c_str(), target, sizeof(target)-1) > 0)
                    out.push_back(target);
            }
        }
        closedir(d);
        return out;
    };

    for (auto &fname: getProcCandidates())
    {
        hid_t file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id >= 0 && H5Lexists(file_id, dataset.c_str(), H5P_DEFAULT ) > 0) {
            *handle_from_procfs = true;
            return file_id;
        }
        H5Fclose(file_id);
    }
    fprintf(stderr, "Failed to identify underlying HDF5 file\n");
    return (hid_t) -1;
}

std::vector<DatasetInfo> readHdf5Datasets(hid_t file_id, std::vector<std::string> &names)
{
    auto readHdf5Dataset = [&](hid_t file_id, std::string dname)
    {
        Benchmark benchmark;
        DatasetInfo out;

        /* Open .h5 file in read-only mode */
        hid_t dset_id = H5Dopen(file_id, dname.data(), H5P_DEFAULT);
        if (dset_id < 0)
        {
            fprintf(stderr, "Failed to open dataset for reading\n");
            return out;
        }

        /* Retrieve datatype */
        out.hdf5_datatype = H5Dget_type(dset_id);
        out.datatype = out.getDatatype();

        /* Retrieve number of dimensions and compute total grid size, in bytes */
        hid_t space_id = H5Dget_space(dset_id);
        out.dimensions.resize(H5Sget_simple_extent_ndims(space_id));
        H5Sget_simple_extent_dims(space_id, out.dimensions.data(), NULL);
        hsize_t n_elements = std::accumulate(
            std::begin(out.dimensions), std::end(out.dimensions), 1, std::multiplies<hsize_t>());

        /* Allocate enough memory so we can read this dataset */
        void *rdata = (void *) malloc(n_elements * H5Tget_size(out.hdf5_datatype));
        if (! rdata)
        {
            fprintf(stderr, "Not enough memory while allocating room for dataset\n");
            return out;
        }

        /* Read the dataset */
        if (H5Dread(dset_id, out.hdf5_datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata) < 0)
        {
            fprintf(stderr, "Failed to read HDF5 dataset\n");
            free(rdata);
            rdata = NULL;
        }
        else
        {
            benchmark.print("Time to read dataset from disk");
        }
        H5Sclose(space_id);
        H5Dclose(dset_id);

        out.name = dname;
        out.data = rdata;
        return out;
    };

    std::vector<DatasetInfo> out;
    for (auto name: names)
    {
        auto info = readHdf5Dataset(file_id, name);
        if (info.data == NULL) {
            fprintf(stderr, "Failed to read input dataset %s from HDF5 file\n", name.c_str());
            for (auto &entry: out)
                free(entry.data);
            out.clear();
            break;
        }
        out.push_back(info);
    }
    return out;
}

void udf(
    std::vector<DatasetInfo> &input_datasets, DatasetInfo &output_dataset,
    size_t *buf_size, void **buf, size_t &nbytes, char *bytecode, long lSize, const char *dtype)
{
    Benchmark benchmark;
    auto result = output_dataset.data;
    auto n_elements = output_dataset.getGridSize();
    auto storage_size = output_dataset.getStorageSize();

    bool success = callLua(input_datasets, output_dataset, bytecode, lSize, dtype);
    if (! success)
    {
        nbytes = 0;
    } 
    else 
    {
        free(*buf);
        *buf = (void *) result;
        *buf_size = n_elements * storage_size;
        nbytes = n_elements * storage_size;
        benchmark.print("Call to user-defined function");
    }
}

static size_t
H5Z_udf_filter_callback(unsigned int flags, size_t cd_nelmts,
const unsigned int *cd_values, size_t nbytes, size_t *buf_size, void **buf)
{
    if (flags & H5Z_FLAG_REVERSE)
    {
        std::string json_string((const char *)*buf);
        json jas = json::parse(json_string);

        /* Retrieve metadata stored in the JSON payload */
        int lSize = jas["lua_bytecode_size"].get<int>();
        auto names = jas["input_datasets"].get<std::vector<std::string>>();
        auto datatype = jas["output_datatype"].get<std::string>();
        auto resolution = jas["output_resolution"].get<std::vector<hsize_t>>();
        auto output_name = jas["output_dataset"].get<std::string>();

        /* Workaround for lack of API to retrieve the HDF5 file handle from the filter callback */
        bool handle_from_procfs = false;
        hid_t file_id = getDatasetHandle(output_name, &handle_from_procfs);
        if (file_id == -1)
            return 0;

        /* Allocate input and output grids */
        auto input_datasets = readHdf5Datasets(file_id, names);
        DatasetInfo output_dataset(output_name, resolution, datatype);
        output_dataset.hdf5_datatype = output_dataset.getHdf5Datatype();
        output_dataset.data = (void *) malloc(
            output_dataset.getStorageSize() * output_dataset.getGridSize());
        if (! output_dataset.data)
        {
            fprintf(stderr, "Not enough memory allocating output grid\n");
            return 0;
        }

        /* Invoke the Lua interpreter to execute the user-defined function */
        auto dtype = output_dataset.getCastDatatype();
        char *bytecode = (char *)(((char *) *buf) + *buf_size - lSize);
        udf(input_datasets, output_dataset, buf_size, buf, nbytes, bytecode, lSize, dtype);

        /* Release memory used by auxiliary datasets */
        for (size_t i=0; i<input_datasets.size(); ++i)
            free(input_datasets[i].data);

        if (handle_from_procfs)
            H5Fclose(file_id);
    }
    else
    {
        std::string json_string((const char *) *buf);
        json jas = json::parse(json_string);
        int lSize = jas["lua_bytecode_size"].get<int>();
        nbytes = json_string.length() + lSize + 1;
        *buf_size = nbytes;
    }

    return nbytes;
}

const H5Z_class2_t H5Z_UDF_FILTER[1] = {{
    H5Z_CLASS_T_VERS,
    HDF5_UDF_FILTER_ID,
    1, 1,
    "hdf5_udf_filter",
    NULL, /* can_apply */
    NULL, /* set_local */
    H5Z_udf_filter_callback,
}};

H5PL_type_t H5PLget_plugin_type(void) { return H5PL_TYPE_FILTER; }
const void *H5PLget_plugin_info(void) { return H5Z_UDF_FILTER; }
