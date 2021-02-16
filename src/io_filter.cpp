/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: hdf5-udf.cpp
 *
 * HDF5 filter callbacks and main interface with the backends.
 */
#include <dirent.h>
#include <H5PLextern.h>
#include <hdf5.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <fstream>

#include "io_filter.h"
#include "dataset.h"
#include "backend.h"
#include "debug.h"
#include "user_profile.h"
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

struct DatasetHandle {
    DatasetHandle(hid_t dset, hid_t space) :
        dset_id(dset),
        space_id(space)
    {
    }

    ~DatasetHandle() {
        if (space_id >= 0)
            H5Sclose(space_id);
        if (dset_id >= 0)
            H5Dclose(dset_id);
    }
    hid_t dset_id;
    hid_t space_id;
};

void releaseHdf5Datasets(
    std::vector<DatasetHandle *> &handles, std::vector<DatasetInfo> &info);

std::string getFilterPath()
{
    std::vector<std::string> paths;

    const char *env = getenv("HDF5_PLUGIN_PATH");
    if (env)
    {
        std::istringstream ss(env);
        std::copy(std::istream_iterator<std::string>(ss),
            std::istream_iterator<std::string>(),
            std::back_inserter(paths));
    }
    else
    {
        paths.push_back("/usr/local/hdf5/lib/plugin");
    }
    for (auto &path: paths) {
        struct stat statbuf;
        auto p = path + "/libhdf5-udf.so";
        if (stat(p.c_str(), &statbuf) == 0)
            return p;
    }
    return "";
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
        // Disable error messages while we probe the candidate file
        H5E_auto2_t old_func;
        void *old_client_data;
        H5Eget_auto2(H5E_DEFAULT, &old_func, &old_client_data);
        H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

        hid_t file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

        // Enable error messages again
        H5Eset_auto(H5E_DEFAULT, old_func, old_client_data);

        if (file_id >= 0 && H5Lexists(file_id, dataset.c_str(), H5P_DEFAULT ) > 0) {
            *handle_from_procfs = true;
            return file_id;
        }
        if (file_id >= 0)
            H5Fclose(file_id);
    }
    fprintf(stderr, "Failed to identify underlying HDF5 file\n");
    return (hid_t) -1;
}

bool readHdf5Datasets(
    hid_t file_id,
    std::vector<std::string> &input_names,
    std::vector<std::string> &scratch_names,
    std::vector<DatasetHandle *> &out_handles,
    std::vector<DatasetInfo> &out_info)
{
    auto readHdf5Dataset = [&](hid_t file_id, std::string dname, bool read_data)
    {
        Benchmark benchmark;

        /* Open .h5 file in read-only mode */
        hid_t dset_id = H5Dopen(file_id, dname.data(), H5P_DEFAULT);
        if (dset_id < 0)
        {
            fprintf(stderr, "Failed to open dataset for reading\n");
            return false;
        }

        /* Set space id */
        hid_t space_id = H5Dget_space(dset_id);

        /* Store dataset dimensions in a vector object and compute total grid size, in bytes */
        std::vector<hsize_t> dims;
        dims.resize(H5Sget_simple_extent_ndims(space_id));
        H5Sget_simple_extent_dims(space_id, dims.data(), NULL);
        hsize_t n_elements = std::accumulate(
            std::begin(dims), std::end(dims), 1, std::multiplies<hsize_t>());

        /* Retrieve datatype */
        hid_t hdf5_datatype = H5Dget_type(dset_id);
        auto datatype_name = getDatatypeName(hdf5_datatype);
        DatasetInfo info(dname, dims, datatype_name, hdf5_datatype);

        /* Allocate enough memory so we can read this dataset */
        void *rdata = (void *) malloc(n_elements * H5Tget_size(info.hdf5_datatype));
        if (! rdata)
        {
            fprintf(stderr, "Not enough memory while allocating room for dataset\n");
            return false;
        }

        /* Our dataset handle deallocator */
        auto handle = new DatasetHandle(dset_id, space_id);

        /* Read the dataset */
        if (read_data)
        {
            if (H5Dread(handle->dset_id, info.hdf5_datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata) < 0)
            {
                fprintf(stderr, "Failed to read HDF5 dataset\n");
                free(rdata);
                delete handle;
                return false;
            }
            benchmark.print("Time to read dataset from disk");
        }
        else
            memset(rdata, 0, n_elements * H5Tget_size(info.hdf5_datatype));

        info.data = rdata;
        out_info.push_back(std::move(info));
        out_handles.push_back(handle);
        return true;
    };

    std::vector<DatasetHandle *> out;
    for (auto name: input_names)
        if (readHdf5Dataset(file_id, name, true) == false)
        {
            fprintf(stderr, "Failed to read input dataset %s from HDF5 file\n", name.c_str());
            releaseHdf5Datasets(out_handles, out_info);
            return false;
        }
    for (auto name: scratch_names)
        if (readHdf5Dataset(file_id, name, false) == false)
        {
            fprintf(stderr, "Failed to allocate scratch dataset for %s\n", name.c_str());
            releaseHdf5Datasets(out_handles, out_info);
            return false;
        }
    return true;
}

void releaseHdf5Datasets(std::vector<DatasetHandle *> &handles, std::vector<DatasetInfo> &info)
{
    for (auto &entry: handles)
        delete entry;
    for (auto &entry: info)
        free(entry.data);
    handles.clear();
    info.clear();
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
        int bytecode_size = jas["bytecode_size"].get<int>();
        auto input_names = jas["input_datasets"].get<std::vector<std::string>>();
        auto scratch_names = jas["scratch_datasets"].get<std::vector<std::string>>();
        auto datatype = jas["output_datatype"].get<std::string>();
        auto resolution = jas["output_resolution"].get<std::vector<hsize_t>>();
        auto output_name = jas["output_dataset"].get<std::string>();
        auto backend_name = jas["backend"].get<std::string>();

        KeyChecks kc;
        if(kc.validate_key() == -1){
            fprintf(stderr, "Error validating user profile\n");
            return 0;
        }

        auto backend = getBackendByName(backend_name);
        if (! backend)
        {
            fprintf(stderr, "No backend has been found to execute %s code\n",
                backend_name.c_str());
            return 0;
        }

        auto filterpath = getFilterPath();
        if (filterpath.size() == 0)
        {
            fprintf(stderr, "Failed to identify path to HDF5-UDF filter\n");
            return 0;
        }

        /* Workaround for lack of API to retrieve the HDF5 file handle from the filter callback */
        bool handle_from_procfs = false;
        hid_t file_id = getDatasetHandle(output_name, &handle_from_procfs);
        if (file_id == -1)
            return 0;

        /* Allocate room for input data */
        std::vector<DatasetHandle *> input_handles;
        std::vector<DatasetInfo> input_info;
        auto ok = readHdf5Datasets(
            file_id, input_names, scratch_names, input_handles, input_info);
        if (! ok)
        {
            fprintf(stderr, "Failed to process input/scratch datasets\n");
            if (handle_from_procfs)
                H5Fclose(file_id);
            return 0;
        }

        /*
         * Allocate room for output data. String and compound datatypes need to
         * have their size properly set so that getStorage() can return accurate
         * information.
         */
        ssize_t output_dataset_size = 0;
        auto datatype_h5id = getHdf5Datatype(datatype);
        if (datatype_h5id == H5T_COMPOUND || datatype_h5id == static_cast<size_t>(H5T_C_S1))
        {
            hid_t output_id = H5Dopen(file_id, output_name.c_str(), H5P_DEFAULT);
            hid_t output_datatype = H5Dget_type(output_id);
            output_dataset_size = H5Tget_size(output_datatype);
            datatype_h5id = output_datatype;

            /* output_datatype is closed after the handle is borrowed from DatasetInfo */
            H5Dclose(output_id);
        }

        DatasetInfo output_dataset(output_name, resolution, datatype, datatype_h5id);
        if (H5Tget_class(datatype_h5id) == H5T_COMPOUND ||
            H5Tget_class(datatype_h5id) == H5T_STRING)
        {
            /* Now that the handle has been borrowed we can close it */
            H5Tclose(datatype_h5id);
        }

        if (output_dataset_size)
        {
            if (H5Tset_size(output_dataset.hdf5_datatype, output_dataset_size) < 0)
            {
                fprintf(stderr, "Failed to set dataset %#lx size\n",
                    output_dataset.hdf5_datatype);
                if (handle_from_procfs)
                    H5Fclose(file_id);
                return 0;
            }
        }

        output_dataset.data = (void *) malloc(
            output_dataset.getStorageSize() * output_dataset.getGridSize());
        if (! output_dataset.data)
        {
            fprintf(stderr, "Not enough memory allocating output grid\n");
            releaseHdf5Datasets(input_handles, input_info);
            if (handle_from_procfs)
                H5Fclose(file_id);
            return 0;
        }

        /* Execute the user-defined function */
        Benchmark benchmark;
        auto dtype = output_dataset.getCastDatatype();
        char *bytecode = (char *)(((char *) *buf) + *buf_size - bytecode_size);
        if (! backend->run(
            filterpath, input_info, output_dataset, dtype, bytecode, bytecode_size))
        {
            nbytes = 0;
        } 
        else 
        {
            auto n_elements = output_dataset.getGridSize();
            auto storage_size = output_dataset.getStorageSize();

            free(*buf);
            *buf = (void *) output_dataset.data;
            *buf_size = n_elements * storage_size;
            nbytes = n_elements * storage_size;
            benchmark.print("Call to user-defined function");
        }

        /* Release memory used by auxiliary datasets */
        releaseHdf5Datasets(input_handles, input_info);
        if (handle_from_procfs)
            H5Fclose(file_id);
    }
    else
    {
        std::string json_string((const char *) *buf);
        json jas = json::parse(json_string);
        int bytecode_size = jas["bytecode_size"].get<int>();
        nbytes = json_string.length() + bytecode_size + 1;
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
