/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: main.cpp
 *
 * Compiles the UDF into a Lua bytecode and embeds it as a
 * HDF5 dataset.
 */
#include <map>
#include <fstream>
#include <hdf5.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>

#include "filter_id.h"
#include "lua_parser.h"
#include "dataset.h"
#include "json.hpp"

#define UDF_LUA_NAME "udf.lua"

#ifndef UDF_LUA_PATH
#define UDF_LUA_PATH "/usr/local/share/hdf5-udf/" UDF_LUA_NAME
#endif

using json = nlohmann::json;
using namespace std;

/* Virtual dataset parser */
class DatasetOptionsParser {
public:
    bool parse(std::string text, DatasetInfo &out);
private:
    bool parseName(std::string text, DatasetInfo &out);
    bool parseDimensions(std::string text, DatasetInfo &out);
    bool parseDataType(std::string text, DatasetInfo &out);
};

bool DatasetOptionsParser::parse(std::string text, DatasetInfo &out)
{
    /* format 1: dataset_name
     * format 2: dataset_name:dimensions:datatype */
    if (parseName(text, out) == false)
        return false;
    if (parseDimensions(text, out) == false)
        return false;
    if (parseDataType(text, out) == false)
        return false;
    return true;
}

bool DatasetOptionsParser::parseName(std::string text, DatasetInfo &out)
{
    auto sep = text.find_first_of(":");
    if (sep == string::npos)
        out.name = text;
    else
        out.name = text.substr(0, sep);
    return true;
}

bool DatasetOptionsParser::parseDimensions(std::string text, DatasetInfo &out)
{
    auto sep = text.find_first_of(":");
    if (sep == string::npos)
    {
        /* No dimensions declared in the input string (not an error) */
        return true;
    }

    std::string res(text.substr(sep+1));
    size_t num_dims = std::count(res.begin(), res.end(), 'x') + 1;
    if (num_dims < 1 || num_dims > 3)
    {
        fprintf(stderr, "Error: unsupported number of dimensions (%jd)\n", num_dims);
        return false;
    }
    auto x = res.substr(0, res.find_first_of(":"));
    out.dimensions.push_back(std::stoi(x));
    if (num_dims == 2)
    {
        auto y = res.substr(res.find_first_of("x")+1, res.find_first_of(":"));
        out.dimensions.push_back(std::stoi(y));
    }
    else if (num_dims == 3)
    {
        auto y = res.substr(res.find_first_of("x")+1, res.find_last_of("x"));
        auto z = res.substr(res.find_last_of("x")+1, res.find_first_of(":"));
        out.dimensions.push_back(std::stoi(y));
        out.dimensions.push_back(std::stoi(z));
    }
    return true;
}

bool DatasetOptionsParser::parseDataType(std::string text, DatasetInfo &out)
{
    auto sep = text.find_last_of(":");
    if (sep == string::npos)
    {
        /* No datatype declared in the input string (not an error) */
        return true;
    }
    out.datatype = text.substr(sep+1);
    out.hdf5_datatype = out.getHdf5Datatype();
    if (out.hdf5_datatype < 0)
    {
        fprintf(stderr, "Datatype '%s' is not supported\n", out.datatype.c_str());
        return false;
    }
    return true;
}

/* Check if a dataset exist in a HDF5 file */
bool dataset_exists(std::string filename, std::string name)
{
    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    bool exists = H5Lexists(file_id, name.c_str(), H5P_DEFAULT ) > 0;
    H5Fclose(file_id);
    return exists;
}

/* Get the Lua template file */
std::string lua_template_path(const char *argv0)
{
    /* Look for the file under $(dirname argv0) */
    char tmp[PATH_MAX];
    memset(tmp, 0, sizeof(tmp));
    if (realpath(argv0, tmp) < 0)
    {
        fprintf(stderr, "Error resolving path %s: %s\n", argv0, strerror(errno));
        return "";
    }
    char *sep = strrchr(tmp, '/');
    if (! sep)
    {
        fprintf(stderr, "Error parsing %s: missing / separator\n", argv0);
        return "";
    }
    *(++sep) = '\0';
    size_t left = PATH_MAX - (sep-tmp) - 1;
    if (strlen(UDF_LUA_NAME) + 1 > left)
    {
        fprintf(stderr, "Path component exceeds PATH_MAX\n");
        return "";
    }
    strcat(sep, UDF_LUA_NAME);

    struct stat statbuf;
    if (stat(tmp, &statbuf) == 0)
    {
        /* Lua template found */
        return std::string(tmp);
    }

    /* Fallback: search under the HDF5 plugin directory */
    if (stat(UDF_LUA_PATH, &statbuf) == -1)
    {
        fprintf(stderr, "Failed to access %s: %s\n", UDF_LUA_PATH, strerror(errno));
        return "";
    }
    return std::string(UDF_LUA_PATH);
}

/* Lua to bytecode */
std::string create_bytecode(std::string input, std::string output, const char *argv0)
{
    std::string bytecode;
    std::ifstream ifs(input);
    if (! ifs.is_open())
    {
        fprintf(stderr, "Failed to open %s\n", input.c_str());
        return "";
    }
    std::string inputFileBuffer(
		(std::istreambuf_iterator<char>(ifs)),
        (std::istreambuf_iterator<char>()  ));

    /* Basic check: does the template file exist? */
    std::string template_path = lua_template_path(argv0);
    if (template_path.size() == 0)
    {
        fprintf(stderr, "Failed to find Lua template file\n");
        return "";
    }
    std::ifstream ifstr(template_path);
    std::string udf(
		(std::istreambuf_iterator<char>(ifstr)),
        (std::istreambuf_iterator<char>()    ));

    /* Basic check: is the template string present in the template file? */
    std::string placeholder = "-- user_callback_placeholder";
    auto start = udf.find(placeholder);
    if (start == std::string::npos)
    {
        fprintf(stderr, "Failed to find placeholder string in %s\n", UDF_LUA_PATH);
        return "";
    }

    /* Embed UDF string in the template */
    auto completeCode = udf.replace(start, placeholder.length(), inputFileBuffer);

    /* Compile the code */
    std::ofstream tmpfile;
    char buffer [32];
    sprintf(buffer, "hdf5-udf-XXXXXX");
    if (mkstemp(buffer) < 0){
        fprintf(stderr, "Error creating temporary file.\n");
        return std::string("");
    }
    tmpfile.open (buffer);
    tmpfile << completeCode.data();
    tmpfile.flush();
    tmpfile.close();

    pid_t pid = fork();
    if (pid == 0)
    {
        // Child process
        char *cmd[] = {
            (char *) "luajit",
            (char *) "-O3",
            (char *) "-b",
            (char *) buffer,
            (char *) output.c_str(),
            (char *) NULL
        };
        execvp(cmd[0], cmd);
    }
    else if (pid > 0)
    {
        // Parent
        int exit_status;
        wait4(pid, &exit_status, 0, NULL);

        struct stat statbuf;
        if (stat(output.c_str(), &statbuf) == 0) {
            printf("Bytecode has %ld bytes\n", statbuf.st_size);

            std::ifstream data(output, std::ifstream::binary);
            std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(data), {});
            bytecode.assign(buffer.begin(), buffer.end());

            unlink(output.c_str());
        }
        unlink(buffer);
        return bytecode;
    }
    fprintf(stderr, "Failed to execute luajit\n");
    return bytecode;
}

int main(int argc, char **argv)
{
    if(argc < 3)
    {
        fprintf(stdout,
            "Syntax: %s <hdf5_file> <lua_file> [--overwrite] [virtual_dataset..]\n\n"
            "Options:\n"
            "  hdf5_file                      Input/output HDF5 file\n"
            "  lua_file                       Lua script with user-defined-function\n"
            "  virtual_dataset                Virtual dataset(s) to create. See syntax below.\n"
            "                                 If omitted, dataset names are picked from lua_file\n"
            "                                 and their resolutions/types are set to match the input\n"
            "                                 datasets declared in that same file\n"
            "  --overwrite                    Overwrite existing virtual dataset(s)\n\n"
            "Formatting options for <virtual_dataset>:\n"
            "  dataset_name:resolution:type   dataset_name: name of the virtual dataset\n"
            "                                 resolution: XRES, XRESxYRES, or XRESxYRESxZRES\n"
            "                                 type: [u]int16, [u]int32, [u]int64, float, or double\n\n"
            "Examples:\n"
            "%s sample.h5 simple_vector.lua Simple:500:float\n"
            "%s sample.h5 sine_wave.lua SineWave:100x10:int32\n",
            argv[0], argv[0], argv[0]);
        exit(1);
    }

    /* Sanity checks */
    if (H5Zfilter_avail(HDF5_UDF_FILTER_ID) <= 0)
    {
        fprintf(stderr, "Could not locate the HDF5-UDF filter\n");
        fprintf(stderr, "Make sure to set $HDF5_PLUGIN_PATH prior to running this tool\n");
        exit(1);
    }

    std::string hdf5_file = argv[1];
    std::string lua_file = argv[2];
    const int first_dataset_index = 3;
    bool overwrite = false;

    /* Process virtual (output) datasets given in the command line */
    std::vector<DatasetInfo> virtual_datasets;
    std::vector<std::string> delete_list;
    for (int i=first_dataset_index; i<argc; ++i)
    {
        DatasetInfo info;
        DatasetOptionsParser parser;
        if (strcmp(argv[i], "--overwrite") == 0)
        {
            overwrite = true;
            continue;
        }
        if (parser.parse(argv[i], info) == false)
        {
            fprintf(stderr, "Failed to parse string '%s'\n", argv[i]);
            exit(1);
        }
        if (dataset_exists(hdf5_file, info.name))
        {
            if (overwrite)
                delete_list.push_back(info.name);
            else
            {
                fprintf(stderr, "Error: dataset %s already exists\n", info.name.c_str());
                exit(1);
            }
        }
        virtual_datasets.push_back(info);
    }

    /* Identify virtual dataset name(s) and input dataset(s) that the Lua code depends on */
    std::vector<std::string> dataset_names = LuaParser(lua_file).parseNames();
    std::vector<DatasetInfo> input_datasets;
    for (auto &name: dataset_names)
    {
        DatasetInfo info;
        info.name = name;
        if (dataset_exists(hdf5_file, name))
        {
            /* Open HDF5 file */
            hid_t file_id = H5Fopen(hdf5_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            if (file_id < 0)
            {
                fprintf(stderr, "Error opening %s\n", hdf5_file.c_str());
                exit(1);
            }

            /* Retrieve dataset information */
            hid_t dset_id = H5Dopen(file_id, info.name.c_str(), H5P_DEFAULT);
            if (dset_id < 0)
            {
                fprintf(stderr, "Error opening dataset %s from file %s\n",
                    info.name.c_str(), hdf5_file.c_str());
                exit(1);
            }
            hid_t space_id = H5Dget_space(dset_id);
            int ndims = H5Sget_simple_extent_ndims(space_id);
            info.dimensions.resize(ndims);
            info.hdf5_datatype = H5Dget_type(dset_id);
            H5Sget_simple_extent_dims(space_id, info.dimensions.data(), NULL);

            /* Check that the input dataset's datatype is supported by our implementation */
            auto datatype_ptr = info.getDatatype();
            if (datatype_ptr == NULL) {
                fprintf(stderr, "Unsupported HDF5 datatype %jd\n", info.hdf5_datatype);
                exit(1);
            }
            info.datatype = datatype_ptr;

            input_datasets.push_back(info);
            H5Sclose(space_id);
            H5Dclose(dset_id);
            H5Fclose(file_id);
            info.printInfo("Input");
        }
        else
        {
            /* This is an output (virtual) dataset */
            bool already_given_in_cmdline = false;
            for (auto &entry: virtual_datasets)
                if (entry.name.compare(name) == 0)
                {
                    already_given_in_cmdline = true;
                    break;
                }
            if (! already_given_in_cmdline)
            {
                info.name = name;
                virtual_datasets.push_back(info);
            }
        }
    }

    if (virtual_datasets.size() == 0)
    {
        fprintf(stderr,
            "Error: all datasets given in the Lua script already exist.\n"
            "Please explicitly specify the virtual dataset(s) in the command line.\n");
        exit(1);
    }

    /*
     * Validate output dimensions and datatypes. If some information is not
     * available, pick it up from the input datasets parsed in the loop above.
     */
    for (auto &info: virtual_datasets)
    {
        if (info.hdf5_datatype != -1) {
            info.printInfo("Virtual");
            continue;
        }

        /* Make sure that we have at least one input dataset */
        if (input_datasets.size() == 0)
        {
            fprintf(stderr, "Cannot determine dimensions and type of virtual dataset %s. Please specify.\n",
                info.name.c_str());
            exit(1);
        }

        /* Require that all input datasets have the same dimensions and type */
        for (size_t i=1; i<input_datasets.size(); ++i)
        {
            if (! H5Tequal(input_datasets[i].hdf5_datatype, input_datasets[i-1].hdf5_datatype))
            {
                fprintf(stderr, "Cannot determine type of virtual dataset %s. Please specify.\n",
                    info.name.c_str());
                exit(1);
            }
            if (input_datasets[i].dimensions != input_datasets[i-1].dimensions)
            {
                fprintf(stderr, "Cannot determine dimensions of virtual dataset %s. Please specify.\n",
                    info.name.c_str());
                exit(1);
            }
        }

        /* We're all set: copy attributes from the first input dataset */
        info.hdf5_datatype = input_datasets[0].hdf5_datatype;
        info.datatype = input_datasets[0].datatype;
        info.dimensions = input_datasets[0].dimensions;
        info.printInfo("Virtual");
    }

    /* Generate a bytecode from the Lua source file */
    std::string bytecode_file = lua_file + ".bytecode";
    std::string bytecode = create_bytecode(lua_file, bytecode_file, argv[0]);
    if (bytecode.size() == 0)
    {
        fprintf(stderr, "Failed to create bytecode file\n");
        exit(1);
    }

    /* Create the virtual datasets */
    for (auto &info: virtual_datasets)
    {
        hid_t file_id = H5Fopen(hdf5_file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        if (file_id < 0)
        {
            fprintf(stderr, "Error opening %s\n", hdf5_file.c_str());
            exit(1);
        }

        /* Create dataspace */
        hid_t space_id = H5Screate_simple(info.dimensions.size(), info.dimensions.data(), NULL);
        if (space_id < 0)
        {
            fprintf(stderr, "Failed to create dataspace\n");
            exit(1);
        }

        /* Create virtual dataset creation property list */
        hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
        if (dcpl_id < 0)
        {
            fprintf(stderr, "Failed to create dataset property list\n");
            exit(1);
        }

        herr_t status;
        status = H5Pset_filter(dcpl_id, HDF5_UDF_FILTER_ID, H5Z_FLAG_MANDATORY, 0, NULL);
        if (status < 0)
        {
            fprintf(stderr, "Failed to configure dataset filter\n");
            fprintf(stderr, "Make sure to set $HDF5_PLUGIN_PATH prior to running this tool\n");
            exit(1);
        }

        status = H5Pset_chunk(dcpl_id, info.dimensions.size(), info.dimensions.data());
        if (status < 0)
        {
            fprintf(stderr, "Failed to set chunk size\n");
            exit(1);
        }

        if (std::find(delete_list.begin(), delete_list.end(), info.name) != delete_list.end())
        {
            /* Delete existing dataset so its contents can be overwritten */
            status = H5Ldelete(file_id, info.name.c_str(), H5P_DEFAULT);
            if (status < 0)
            {
                fprintf(stderr, "Failed to delete existing virtual dataset %s\n", info.name.c_str());
                exit(1);
            }
        }

        /* Create virtual dataset */
        hid_t dset_id = H5Dcreate(file_id, info.name.c_str(), info.hdf5_datatype, space_id, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
        if (dset_id < 0)
        {
            fprintf(stderr, "Failed to create dataset\n");
            exit(1);
        }

        /* Prepare data for JSON payload */
        std::vector<std::string> input_dataset_names;
        std::transform(input_datasets.begin(), input_datasets.end(), std::back_inserter(input_dataset_names),
            [](DatasetInfo info) -> std::string { return info.name; });

        /* JSON Payload */
        json jas;
        jas["output_dataset"] = info.name;
        jas["output_resolution"] = info.dimensions;
        jas["output_datatype"] = info.datatype;
        jas["input_datasets"] = input_dataset_names;
        jas["lua_bytecode_size"] = bytecode.length();

        std::string jas_str = jas.dump();
        size_t payload_size = jas_str.length() + bytecode.size() + 1;
        printf("%s dataset header:\n%s\n", info.name.c_str(), jas.dump(4).c_str());

        /* Sanity check: the JSON and the bytecode must fit in the dataset */
        hsize_t grid_size = std::accumulate(std::begin(info.dimensions), std::end(info.dimensions), 1, std::multiplies<hsize_t>());
        if (payload_size > (grid_size * H5Tget_size(info.hdf5_datatype)))
        {
            /* TODO: fallback to saving a regular dataset */
            fprintf(stderr, "Error: len(JSON+bytecode) > virtual dataset dimensions\n");
            exit(1);
        }

        /* Prepare payload data */
        char *payload = (char *) malloc(grid_size * H5Tget_size(info.hdf5_datatype));
        char *p = payload;
        memcpy(p, &jas_str[0], jas_str.length());
        p += jas_str.length();
        *p = '\0';
        memcpy(&p[1], &bytecode[0], bytecode.size());

        /* Write the data to the dataset */
        status = H5Dwrite(dset_id, info.hdf5_datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, payload);
        if (status < 0)
        {
            fprintf(stderr, "Failed to write to the dataset\n");
            exit(1);
        }

        /* Close and release resources */
        status = H5Pclose(dcpl_id);
        status = H5Dclose(dset_id);
        status = H5Sclose(space_id);
        status = H5Fclose(file_id);
        free(payload);
    }
    
    return 0;
}
