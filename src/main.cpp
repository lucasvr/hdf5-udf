/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: main.cpp
 *
 * Compiles the UDF into executable form and embeds it as a
 * HDF5 dataset.
 */
#include <map>
#include <fstream>
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
#include <regex>

#include "filter_id.h"
#include "dataset.h"
#include "backend.h"
#include "json.hpp"

using json = nlohmann::json;

/* Helper functions */
int getUserStringSize(std::string user_type)
{
    /*
     * UDF strings can be defined as 'string' or as 'string(N)'.
     * In the first case the string size is determined by the
     * constant DEFAULT_UDF_STRING_SIZE.
     */
    auto start = user_type.find("(");
    auto end = user_type.find(")");
    if (start == std::string::npos && end == std::string::npos)
        return DEFAULT_UDF_STRING_SIZE;
    else if (start != std::string::npos && end != std::string::npos)
    {
        auto ssize = user_type.substr(start+1, end-start-1);
        return std::stoi(ssize);
    }
    else
    {
        fprintf(stderr, "Invalid string syntax: %s\n", user_type.c_str());
        return -1;
    }
}


/* Virtual dataset parser */
class DatasetOptionsParser {
public:
    bool parse(std::string text, DatasetInfo &out);
private:
    bool parseName(std::string text, DatasetInfo &out);
    bool parseDimensions(std::string text, DatasetInfo &out);
    bool parseDataType(std::string text, DatasetInfo &out, size_t size);
    bool parseCompoundMembers(std::string text, DatasetInfo &out, size_t &size);
    bool is_compound;
};

bool DatasetOptionsParser::parse(std::string text, DatasetInfo &out)
{
    /* format 1: dataset_name
     * format 2: dataset_name:dimensions:datatype
     * format 3: dataset_name:{member:type[,member:type...]}:dimensions */
    size_t compound_size = 0;
    is_compound = text.find_first_of("{") != std::string::npos;
    if (parseName(text, out) == false)
        return false;
    if (parseDimensions(text, out) == false)
        return false;
    if (is_compound && parseCompoundMembers(text, out, compound_size) == false)
        return false;
    if (parseDataType(text, out, compound_size) == false)
        return false;
    return true;
}

bool DatasetOptionsParser::parseName(std::string text, DatasetInfo &out)
{
    auto sep = text.find_first_of(":");
    if (sep == std::string::npos)
        out.name = text;
    else
        out.name = text.substr(0, sep);
    return true;
}

bool DatasetOptionsParser::parseDimensions(std::string text, DatasetInfo &out)
{
    /* On a compound, dimensions are the last declared element */
    auto sep = is_compound ? text.find_last_of(":") : text.find_first_of(":");
    if (sep == std::string::npos)
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
    try
    {
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
    }
    catch (std::invalid_argument &e)
    {
        fprintf(stderr, "Failed to extract dimensions from '%s'\n", text.c_str());
        return false;
    }
    return true;
}

bool DatasetOptionsParser::parseDataType(std::string text, DatasetInfo &out, size_t size)
{
    if (is_compound)
    {
        out.datatype = "compound";
        out.hdf5_datatype = H5Tcreate(H5T_COMPOUND, size);
        for (auto &member: out.members)
        {
            auto type = getHdf5Datatype(member.usertype);
            if (type == static_cast<size_t>(H5T_C_S1))
            {
                type = H5Tcopy(H5T_C_S1);
                auto size = getUserStringSize(member.usertype);
                if (size < 0)
                {
                    fprintf(stderr, "Invalid syntax describing string element in compound\n");
                    H5Tclose(out.hdf5_datatype);
                    H5Tclose(type);
                    return false;
                }
                H5Tset_size(type, size);
            }

            H5Tinsert(out.hdf5_datatype, member.name.c_str(), member.offset, type);

            if (H5Tget_class(type) == H5T_C_S1)
                H5Tclose(type);
        }
    }
    else
    {
        auto sep = text.find_last_of(":");
        if (sep == std::string::npos)
        {
            /* No datatype declared in the input string (not an error) */
            return true;
        }
        out.datatype = text.substr(sep+1);
        out.hdf5_datatype = getHdf5Datatype(out.datatype);
        if (out.hdf5_datatype < 0)
        {
            fprintf(stderr, "Datatype '%s' is not supported\n", out.datatype.c_str());
            return false;
        }
        out.hdf5_datatype = H5Tcopy(out.hdf5_datatype);
    }
    return true;
}

bool DatasetOptionsParser::parseCompoundMembers(std::string text, DatasetInfo &out, size_t &size)
{
    /* Format: dataset_name{member:type[,member:type...]}:dimensions */
    auto start = text.find_first_of("{"), end = text.find_last_of("}");
    if (start == std::string::npos || end == std::string::npos)
    {
        fprintf(stderr, "Invalid syntax to describe a compound dataset\n");
        return false;
    }
    auto memberlist = text.substr(start+1, end-start-1);
    std::regex re("([A-Za-z0-9_]+:[A-Za-z0-9_()]+)");
    std::smatch matches;

    size = 0;
    while (std::regex_search(memberlist, matches, re))
    {
        auto sep = matches.str(0).find_first_of(":");
        auto name = matches.str(0).substr(0, sep);
        auto type = matches.str(0).substr(sep+1);
        std::string cast(getCastDatatype(getHdf5Datatype(type)));

        auto ptr = cast.find("*");
        if (ptr != std::string::npos)
            cast.erase(ptr);

        bool is_string = type.compare(0, 6, "string") == 0;

        CompoundMember member;
        member.name = name;
        member.type = cast;
        member.usertype = type;
        member.offset = size;
        member.size = is_string ?
            getUserStringSize(type) :
            getStorageSize(getHdf5Datatype(type));
        member.is_char_array = is_string;

        if (is_string && member.size < 0)
        {
            fprintf(stderr, "Invalid syntax in compound dataset description\n");
            return false;
        }

        out.members.push_back(member);
        memberlist = matches.suffix().str();
        size += member.size;
    }
    return true;
}

/* Open group. Expects either a simple name as in "ds" or the full path ("/group/ds") */
hid_t open_group(hid_t file_id, std::string path, bool print_errors)
{
    /* Split input path by '/' delimiter, populating @groups */
    std::stringstream input(path);
    std::vector<std::string> groups;
    std::string group_name;

    groups.push_back("/");
    while (std::getline(input, group_name, '/'))
    {
        if (group_name.size())
            groups.push_back(group_name);
    }
    groups.pop_back();

    /* Open intermediate groups */
    hid_t parent_id = file_id;
    std::vector<hid_t> group_ids;
    for (size_t i=0; i<groups.size(); ++i)
    {
        group_ids.push_back(H5Gopen(parent_id, groups[i].c_str(), H5P_DEFAULT));
        if (group_ids.back() < 0)
        {
            if (print_errors)
                fprintf(stderr, "Failed to open group '%s' in '%s'\n",
                    groups[i].c_str(), path.c_str());
            break;
        }
        parent_id = group_ids.back();
    }

    for (size_t i=0; i<group_ids.size()-1; ++i)
        H5Gclose(group_ids[i]);
    return group_ids.back();
}

/* Open dataset. Expects either a simple name as in "ds" or the full path ("/group/ds") */
hid_t open_dataset(hid_t file_id, std::string path, bool print_errors)
{
    hid_t dset_id = -1;
    hid_t group_id = open_group(file_id, path, print_errors);
    if (group_id >= 0)
    {
        auto index = path.find_last_of("/");
        auto dataset_name = index >= 0 ? path.substr(index+1) : path;
        dset_id = H5Dopen(group_id, dataset_name.c_str(), H5P_DEFAULT);
        if (dset_id < 0 && print_errors)
            fprintf(stderr, "Error opening dataset %s\n", path.c_str());
        H5Gclose(group_id);
    }
    return dset_id;
}

/* Check if a dataset exist in a HDF5 file */
bool dataset_exists(std::string filename, std::string name)
{
    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0)
    {
        fprintf(stderr, "Failed to open file %s\n", filename.c_str());
        return false;
    }
    hid_t group_id = open_group(file_id, name, true);
    bool exists = false;
    if (group_id >= 0)
    {
        auto index = name.find_last_of("/");
        auto dataset_name = index >= 0 ? name.substr(index+1) : name;
        exists = H5Lexists(group_id, name.c_str(), H5P_DEFAULT);
        H5Gclose(group_id);
    }
    H5Fclose(file_id);
    return exists;
}

/* Get the template file, if one exists for the given backend */
std::string template_path(std::string backend_extension, std::string argv0)
{
    char dirname_argv0[PATH_MAX], tmp[PATH_MAX];
    memset(dirname_argv0, 0, sizeof(dirname_argv0));
    if (! realpath(argv0.c_str(), dirname_argv0) && ! realpath("/proc/self/exe", dirname_argv0))
    {
        fprintf(stderr, "Error resolving path to '%s': %s\n", argv0.c_str(), strerror(errno));
        return "";
    }
    char *sep = strrchr(dirname_argv0, '/');
    if (! sep)
    {
        fprintf(stderr, "Error parsing %s: missing / separator\n", argv0.c_str());
        return "";
    }
    *(sep) = '\0';

    /* Look for the file under $(dirname argv0) */
    struct stat statbuf;
    int n = snprintf(tmp, sizeof(tmp)-1, "%s/udf_template%s",
        dirname_argv0, backend_extension.c_str());
    if (static_cast<size_t>(n) >= sizeof(tmp)-1)
    {
        fprintf(stderr, "Path component exceeds PATH_MAX\n");
        return "";
    }
    if (stat(tmp, &statbuf) == 0)
        return std::string(tmp);

    /* Look for the file under $(dirname argv0)/../share/hdf5-udf */
    n = snprintf(tmp, sizeof(tmp)-1, "%s/../share/hdf5-udf/udf_template%s",
        dirname_argv0, backend_extension.c_str());
    if (static_cast<size_t>(n) >= sizeof(tmp)-1)
    {
        fprintf(stderr, "Path component exceeds PATH_MAX\n");
        return "";
    }
    if (stat(tmp, &statbuf) == 0)
        return std::string(tmp);

    /* Bad installation or given backend does not provide a template file */
    return "";
}

int main(int argc, char **argv)
{
    if(argc < 3)
    {
        fprintf(stdout,
            "Syntax: %s <hdf5_file> <udf_file> [--overwrite] [virtual_dataset..]\n\n"
            "Options:\n"
            "  hdf5_file                      Input/output HDF5 file\n"
            "  udf_file                       File implementing the user-defined-function\n"
            "  virtual_dataset                Virtual dataset(s) to create. See syntax below.\n"
            "                                 If omitted, dataset names are picked from udf_file\n"
            "                                 and their resolutions/types are set to match the input\n"
            "                                 datasets declared on that same file\n"
            "  --overwrite                    Overwrite existing virtual dataset(s)\n\n"
            "Formatting options for <virtual_dataset>:\n"
            "  name:resolution:type           name:       name of the virtual dataset\n"
            "                                 resolution: XRES, XRESxYRES, or XRESxYRESxZRES\n"
            "                                 type:       [u]int8, [u]int16, [u]int32, [u]int64, float,\n"
            "                                             double, string, or string(NN)\n"
            "                                 If unset, strings have a fixed size of 32 characters.\n\n"
            "Formatting options for outputting compound datasets:\n"
            "  name:{member:type[,member:type...]}:resolution\n\n"
            "Examples:\n"
            "%s sample.h5 simple_vector.lua Simple:500:float\n"
            "%s sample.h5 sine_wave.lua SineWave:100x10:int32\n"
            "%s sample.h5 string_generator.lua 'Words:1000:string(80)'\n"
            "%s sample.h5 virtual.py /Group/Name/VirtualDataset:100x100:uint8\n"
            "%s sample.h5 compound.cpp 'Observations:{id:uint8,location:string,temperature:float}:1000'\n\n",
            argv[0], argv[0], argv[0], argv[0], argv[0], argv[0]);
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
    std::string udf_file = argv[2];
    const int first_dataset_index = 3;
    bool overwrite = false;

    struct stat statbuf;
    std::vector<std::string> input_files = {hdf5_file, udf_file};
    for (auto input_file: input_files)
        if (stat(input_file.c_str(), &statbuf) < 0)
        {
            fprintf(stderr, "%s: %s\n", input_file.c_str(), strerror(errno));
            exit(1);
        }

    Backend *backend = getBackendByFileExtension(udf_file);
    if (! backend)
    {
        fprintf(stderr, "Could not identify a parser for %s\n", udf_file.c_str());
        exit(1);
    }
    printf("Backend: %s\n", backend->name().c_str());

    /* Process virtual (output) datasets given in the command line */
    std::vector<DatasetInfo> virtual_datasets;
    std::vector<std::string> delete_list;
    for (int i=first_dataset_index; i<argc; ++i)
    {
        DatasetInfo info("", std::vector<hsize_t>(), "", -1);
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
        virtual_datasets.push_back(std::move(info));
    }

    /* Identify virtual dataset name(s) and input dataset(s) that the UDF code depends on */
    std::vector<std::string> dataset_names = backend->udfDatasetNames(udf_file);
    std::vector<DatasetInfo> input_datasets;
    std::string compound_declarations = "";
    for (auto &name: dataset_names)
    {
        /* If this dataset is scheduled for removal, then assume it's not an input dataset */
        bool is_valid_candidate = true;
        for (auto &deletename: delete_list)
            if (name.compare(deletename) == 0)
            {
                is_valid_candidate = false;
                break;
            }

        if (dataset_exists(hdf5_file, name) && is_valid_candidate)
        {
            /* Open HDF5 file */
            hid_t file_id = H5Fopen(hdf5_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            if (file_id < 0)
            {
                fprintf(stderr, "Error opening %s\n", hdf5_file.c_str());
                exit(1);
            }

            /* Retrieve dataset information */
            hid_t dset_id = open_dataset(file_id, name, true);
            if (dset_id < 0)
            {
                fprintf(stderr, "Error opening dataset %s from file %s\n",
                    name.c_str(), hdf5_file.c_str());
                exit(1);
            }
            hid_t space_id = H5Dget_space(dset_id);
            hid_t hdf5_datatype = H5Dget_type(dset_id);

            int ndims = H5Sget_simple_extent_ndims(space_id);
            std::vector<hsize_t> dimensions(ndims);
            H5Sget_simple_extent_dims(space_id, dimensions.data(), NULL);

            auto datatype_name = getDatatypeName(hdf5_datatype);
            if (datatype_name == NULL)
            {
                fprintf(stderr, "Unsupported HDF5 datatype %#lx\n", hdf5_datatype);
                exit(1);
            }

            /* DatasetInfo entry we'll want to append */
            DatasetInfo info(name, dimensions, datatype_name, hdf5_datatype);

            /* Check that the input dataset's datatype is supported by our implementation */
            if (strcmp(datatype_name, "compound") == 0)
            {
                info.members = getCompoundMembers(hdf5_datatype);
                if (info.members.size() == 0)
                {
                    fprintf(stderr, "Failed to parse dataset %s from file %s\n",
                        name.c_str(), hdf5_file.c_str());
                    exit(1);
                }
                if (compound_declarations.size())
                    compound_declarations += "\n";
                compound_declarations += backend->compoundToStruct(info, false);
            }
            else if (strcmp(datatype_name, "string") == 0)
            {
                // Because HDF5 has both variable- and fixed-sized strings there
                // are two possible ways to define a string element: (1) with
                // 'char varname[size]' and (2) with 'char *varname'. Here we embed
                // string variables into packed structures so that UDFs can easily
                // iterate over them with for loops and basic pointer arithmetic.
                bool is_varstring = H5Tis_variable_str(hdf5_datatype);
                size_t member_size = H5Tget_size(hdf5_datatype);
                info.members.push_back(info.getStringDeclaration(is_varstring, member_size));
                if (compound_declarations.size())
                    compound_declarations += "\n";
                compound_declarations += backend->compoundToStruct(info, true);
            }

            info.printInfo("Input");
            input_datasets.push_back(std::move(info));
            H5Sclose(space_id);
            H5Dclose(dset_id);
            H5Fclose(file_id);
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
                DatasetInfo info(name, std::vector<hsize_t>(), "", -1);
                virtual_datasets.push_back(std::move(info));
            }
        }
    }

    if (virtual_datasets.size() == 0)
    {
        fprintf(stderr,
            "Error: all datasets given in the UDF file already exist.\n"
            "Please explicitly specify the virtual dataset(s) in the command line.\n");
        exit(1);
    }

    /*
     * Validate output dimensions and datatypes. If some information is not
     * available, pick it up from the input datasets parsed in the loop above.
     */
    for (auto &info: virtual_datasets)
    {
        // XXX: path tested
        if (info.datatype.compare(0, 6, "string") == 0)
        {
           /*
            * Ideally, we'd like to  output string datatypes using variable size;
            * that would make our life much easier. However, using variable string
            * sizes means that H5Dwrite() needs to call strlen() on each member of
            * the write data, and that fails badly because we're using the write
            * data buffer to hold the UDF bytecode -- so that call to strlen() is
            * likely to crash the application.
            */
            int string_size = getUserStringSize(info.datatype);
            if (string_size < 0)
            {
                fprintf(stderr, "Failed to parse string size in '%s'\n", info.datatype.c_str());
                exit(1);
            }
            if (H5Tset_size(info.hdf5_datatype, string_size) < 0)
            {
                fprintf(stderr, "Failed to set dataset %#lx to variable size\n", info.hdf5_datatype);
                exit(1);
            }

            /*
             * Embed the output variable in a packed structure so that the
             * UDFs can easily iterate over its members with for loops and
             * basic pointer arithmetic.
             */
            info.members.push_back(
                info.getStringDeclaration(false, string_size));
            if (compound_declarations.size())
                compound_declarations += "\n";
            compound_declarations += backend->compoundToStruct(info, true);
        }
        else if (info.datatype.compare("compound") == 0)
        {
            /*
             * Embed the output variable in a packed structure so that the
             * UDFs can easily iterate over its members with for loops and
             * basic pointer arithmetic.
             */
            if (compound_declarations.size())
                compound_declarations += "\n";
            compound_declarations += backend->compoundToStruct(info, false);
        }

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
        info.hdf5_datatype = H5Tcopy(input_datasets[0].hdf5_datatype);
        info.datatype = input_datasets[0].datatype;
        info.dimensions = input_datasets[0].dimensions;
        info.printInfo("Virtual");
    }

    /* Compile the UDF source file */
    std::vector<DatasetInfo> datasets(input_datasets);
    datasets.insert(datasets.end(), virtual_datasets.begin(), virtual_datasets.end());
    auto template_file = template_path(backend->extension(), argv[0]);
    auto bytecode = backend->compile(udf_file, template_file, compound_declarations, datasets);
    if (bytecode.size() == 0)
    {
        fprintf(stderr, "Failed to compile UDF file\n");
        exit(1);
    }

    /* Delete datasets that we'll overwrite next */
    if (delete_list.size())
    {
        hid_t file_id = H5Fopen(hdf5_file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        if (file_id < 0)
        {
            fprintf(stderr, "Error opening %s\n", hdf5_file.c_str());
            exit(1);
        }
        hid_t group_id = open_group(file_id, hdf5_file, true);
        if (group_id < 0)
        {
            fprintf(stderr, "Unable to find path to %s\n", hdf5_file.c_str());
            exit(1);
        }
        for (auto &info: delete_list)
        {
            /* Delete existing dataset so its contents can be overwritten */
            herr_t status = H5Ldelete(group_id, info.c_str(), H5P_DEFAULT);
            if (status < 0)
            {
                fprintf(stderr, "Failed to delete existing virtual dataset %s\n", info.c_str());
                exit(1);
            }
        }
        H5Gclose(group_id);
        H5Fclose(file_id);
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

        /* Prepare data for JSON payload */
        std::vector<std::string> input_dataset_names, scratch_dataset_names;
        std::transform(input_datasets.begin(), input_datasets.end(), std::back_inserter(input_dataset_names),
            [](DatasetInfo &info) -> std::string { return info.name; });

        for (auto &other: virtual_datasets)
            if (other.name.compare(info.name) != 0)
                scratch_dataset_names.push_back(other.name);

        /* Remove the output string size (if given) from the datatype */
        auto payload_datatype = info.datatype;
        auto sep = payload_datatype.find("(");
        if (sep != std::string::npos)
            payload_datatype = payload_datatype.substr(0, sep);

        /* JSON Payload */
        json jas;
        jas["output_dataset"] = info.name;
        jas["output_resolution"] = info.dimensions;
        jas["output_datatype"] = payload_datatype;
        jas["input_datasets"] = input_dataset_names;
        jas["scratch_datasets"] = scratch_dataset_names;
        jas["bytecode_size"] = bytecode.length();
        jas["backend"] = backend->name();
        jas["api_version"] = 2;

        std::string jas_str = jas.dump();
        size_t payload_size = jas_str.length() + bytecode.size() + 1;
        printf("\n%s dataset header:\n%s\n", info.name.c_str(), jas.dump(4).c_str());
        if (compound_declarations.size())
            printf("\nData structures available to the UDF:\n%s\n", compound_declarations.c_str());

        /* Sanity check: the JSON and the bytecode must fit in the dataset */
        hsize_t grid_size = std::accumulate(std::begin(info.dimensions), std::end(info.dimensions), 1, std::multiplies<hsize_t>());
        if (payload_size > (grid_size * H5Tget_size(info.hdf5_datatype)))
        {
            /* TODO: fallback to saving a regular dataset */
            fprintf(stderr, "Error: len(JSON+bytecode) > virtual dataset dimensions\n");
            exit(1);
        }

        /* Prepare payload data */
        char *payload = (char *) calloc(grid_size, H5Tget_size(info.hdf5_datatype));
        char *p = payload;
        memcpy(p, &jas_str[0], jas_str.length());
        p += jas_str.length();
        *p = '\0';
        memcpy(&p[1], &bytecode[0], bytecode.size());

        /* Create virtual dataset */
        hid_t dset_id = H5Dcreate(file_id, info.name.c_str(), info.hdf5_datatype, space_id, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
        if (dset_id < 0)
        {
            fprintf(stderr, "Failed to create dataset\n");
            exit(1);
        }

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
