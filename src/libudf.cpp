/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: libudf.cpp
 *
 * Public C/C++ API
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <hdf5.h>

#include <algorithm>
#include <regex>
#include <vector>
#include <string>
#include <sstream>

#include "user_profile.h"
#include "io_filter.h"
#include "hdf5-udf.h"
#include "dataset.h"
#include "backend.h"
#include "json.hpp"

using json = nlohmann::json;

#define INFO(msg...) printf(msg)

#define FAIL(msg...) do { \
    fprintf(stderr, "Error: "); fprintf(stderr, msg); fprintf(stderr, "\n"); \
    return false; \
} while(0)

#define FAIL_INT(msg...) do { \
    fprintf(stderr, "Error: "); fprintf(stderr, msg); fprintf(stderr, "\n"); \
    return -1; \
} while(0)

#define FAIL_STR(msg...) do { \
    fprintf(stderr, "Error: "); fprintf(stderr, msg); fprintf(stderr, "\n"); \
    return ""; \
} while(0)

#define CHECK_ARG_PTR(arg) do { \
    if (arg == NULL) { \
        fprintf(stderr, "%s: ", __func__); \
        FAIL("NULL argument '" #arg "'"); \
    } \
} while(0)

#define EXPORT extern "C"

//////////////////////////////////////////////////////////////////
// Opaque structure that holds the data needed by the public API
//////////////////////////////////////////////////////////////////

typedef struct udf_context {
    udf_context(std::string hdf5_file_path, std::string udf_file_path) :
        hdf5_file(hdf5_file_path),
        udf_file(udf_file_path),
        backend(NULL),
        needs_overwriting(false) { }

    ~udf_context() {
        delete backend;
    }

    // User-provided information
    std::string hdf5_file;
    std::string udf_file;
    std::map<std::string, std::string> options;

    // Generated and managed by libudf
    Backend *backend;
    std::string compound_declarations;
    std::vector<DatasetInfo> input_datasets;
    std::vector<DatasetInfo> udf_datasets;
    std::string sourcecode;
    std::string bytecode;
    bool needs_overwriting;
} udf_context;

/////////////////////
// Helper functions
/////////////////////

static bool libudf_scan(udf_context *ctx);
static bool libudf_validate_udf_datasets(udf_context *ctx);

static int getUserStringSize(std::string user_type)
{
     // UDF strings can be defined as 'string' or as 'string(N)'.
     // In the first case the string size is determined by the
     // constant DEFAULT_UDF_STRING_SIZE.
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
        FAIL_INT("invalid string syntax: %s", user_type.c_str());
}

static bool udfDatasetAlreadySeen(std::string name, udf_context *ctx)
{
    for (auto &entry: ctx->udf_datasets)
        if (entry.name.compare(name) == 0)
            return true;
    return false;
}

static bool isInputDataset(std::string name, udf_context *ctx)
{
    return !udfDatasetAlreadySeen(name, ctx);
}

//////////////////////
// HDF5 file handler
//////////////////////

class HDF5_Handler {
public:
    HDF5_Handler(std::string hdf5_file_path) :
        hdf5_file(hdf5_file_path),
        file_id(-1),
        dset_id(-1),
        space_id(-1),
        dcpl_id(-1),
        hdf5_datatype(-1),
        datatype_name(NULL) { }

    ~HDF5_Handler() {
        if (dcpl_id >= 0)
            H5Pclose(dcpl_id);
        if (space_id >= 0)
            H5Sclose(space_id);
        if (dset_id >= 0)
            H5Dclose(dset_id);
        if (file_id >= 0)
            H5Fclose(file_id);
    }

    // Open the HDF5 file.
    bool openFile(int mode);

    // Open a group by name. Both simple names and full paths are allowed (e.g., "ds", "/group/ds")
    hid_t openGroup(std::string path, bool print_errors);

    // Open a dataset by name. Both simple names and full paths are allowed (e.g., "ds", "/group/ds")
    bool openDataset(std::string path, bool print_errors);

    // Check if a dataset exist in a HDF5 file
    bool hasDataset(std::string name);

    // Attach a payload as a new UDF dataset
    bool createUserDefinedDataset(std::string name, hid_t datatype, const void *payload);

    // Delete a list of datasets from a HDF5 file
    bool deleteDatasets(const std::vector<std::string> &datasets);

    // Configure the UDF I/O filter: number of dimensions and their sizes
    bool configureFilter(int rank, const hsize_t *dims);

    // Extract metadata of an open dataset: number of dimensions, data type, etc.
    bool extractInfo();

    std::string hdf5_file;
    hid_t file_id;
    hid_t dset_id;
    hid_t space_id;
    hid_t dcpl_id;
    hid_t hdf5_datatype;
    const char *datatype_name;
    std::vector<hsize_t> dimensions;
};

bool HDF5_Handler::openFile(int mode)
{
    file_id = H5Fopen(hdf5_file.c_str(), mode, H5P_DEFAULT);
    if (file_id < 0)
        FAIL("could not open %s", hdf5_file.c_str());
    return true;
}

hid_t HDF5_Handler::openGroup(std::string path, bool print_errors)
{
    // Split input path by '/' delimiter, populating @groups
    std::stringstream input(path);
    std::vector<std::string> groups;
    std::string group_name;

    groups.push_back("/");
    while (std::getline(input, group_name, '/'))
        if (group_name.size())
            groups.push_back(group_name);
    groups.pop_back();

    // Open intermediate groups
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

bool HDF5_Handler::openDataset(std::string path, bool print_errors)
{
    hid_t group_id = openGroup(path, print_errors);
    if (group_id >= 0)
    {
        auto index = path.find_last_of("/");
        auto dataset_name = index != path.npos ? path.substr(index+1) : path;
        dset_id = H5Dopen(group_id, dataset_name.c_str(), H5P_DEFAULT);
        H5Gclose(group_id);
        if (dset_id < 0 && print_errors)
            FAIL("can't open dataset %s", path.c_str());
    }
    return true;
}

bool HDF5_Handler::hasDataset(std::string name)
{
    hid_t group_id = openGroup(name, true);
    bool exists = false;
    if (group_id >= 0)
    {
        auto index = name.find_last_of("/");
        auto dataset_name = index != name.npos ? name.substr(index+1) : name;
        exists = H5Lexists(group_id, dataset_name.c_str(), H5P_DEFAULT);
        H5Gclose(group_id);
    }
    return exists;
}

bool HDF5_Handler::createUserDefinedDataset(std::string name, hid_t datatype, const void *payload)
{
    dset_id = H5Dcreate(file_id, name.c_str(), datatype, space_id, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    if (dset_id < 0)
        FAIL("failed to create dataset");

    /* Write the data to the dataset */
    herr_t status = H5Dwrite(dset_id, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, payload);
    if (status < 0)
        FAIL("failed to write UDF dataset to file");

    return true;
}

bool HDF5_Handler::deleteDatasets(const std::vector<std::string> &datasets)
{
    for (auto &dataset: datasets)
    {
        hid_t group_id = openGroup(dataset, true);
        if (group_id < 0)
            FAIL("unable to open group of dataset %s", dataset.c_str());

        herr_t status = H5Ldelete(group_id, dataset.c_str(), H5P_DEFAULT);
        if (status < 0)
        {
            H5Gclose(group_id);
            FAIL("failed to delete dataset %s", dataset.c_str());
        }
        H5Gclose(group_id);
    }
    return true;
}

bool HDF5_Handler::configureFilter(int rank, const hsize_t *dims)
{
    space_id = H5Screate_simple(rank, dims, NULL);
    if (space_id < 0)
        FAIL("failed to create dataspace");

    dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    if (dcpl_id < 0)
        FAIL("failed to create dataset property list");

    herr_t status;
    status = H5Pset_filter(dcpl_id, HDF5_UDF_FILTER_ID, H5Z_FLAG_MANDATORY, 0, NULL);
    if (status < 0)
        FAIL("failed to configure I/O filter. Please check that $HDF5_PLUGIN_PATH is set.");

    status = H5Pset_chunk(dcpl_id, rank, dims);
    if (status < 0)
        FAIL("failed to set chunk size\n");

    return true;
}

bool HDF5_Handler::extractInfo()
{
    space_id = H5Dget_space(dset_id);
    dimensions.resize(H5Sget_simple_extent_ndims(space_id));
    H5Sget_simple_extent_dims(space_id, dimensions.data(), NULL);

    hdf5_datatype = H5Dget_type(dset_id);
    datatype_name = getDatatypeName(hdf5_datatype);
    if (datatype_name == NULL)
        FAIL("unsupported HDF5 datatype %#lx", (long) hdf5_datatype);

    return true;
}

///////////////////////
// UDF dataset parser
///////////////////////

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

/////////////////////////////////////
// Implementation of the public API
/////////////////////////////////////

EXPORT udf_context *libudf_init(const char *hdf5_file, const char *udf_file)
{
    // Check availability of the I/O filter
    if (H5Zfilter_avail(HDF5_UDF_FILTER_ID) <= 0)
    {
        fprintf(stderr,
            "Could not locate the HDF5-UDF filter\n"
            "Make sure to set $HDF5_PLUGIN_PATH prior to running this tool\n");
        return NULL;
    }
    return new udf_context(hdf5_file, udf_file);
}

EXPORT void libudf_destroy(udf_context *ctx)
{
    delete ctx;
}

EXPORT bool libudf_set_option(const char *option, const char *value, udf_context *ctx)
{
    CHECK_ARG_PTR(option);
    CHECK_ARG_PTR(value);
    CHECK_ARG_PTR(ctx);

    // No validation for now.
    ctx->options[option] = value;
    return true;
}

EXPORT bool libudf_push_dataset(const char *description, udf_context *ctx)
{
    CHECK_ARG_PTR(description);
    CHECK_ARG_PTR(ctx);

    DatasetInfo info("", std::vector<hsize_t>(), "", -1);
    info.is_input_dataset = false;

    DatasetOptionsParser parser;
    if (parser.parse(description, info) == false)
        FAIL("failed to parse string '%s'", description);

    HDF5_Handler h5(ctx->hdf5_file);
    if (h5.openFile(H5F_ACC_RDONLY) == false)
        return false;
    else if (h5.hasDataset(info.name))
    {
        auto opt = ctx->options.find("overwrite");
        if (opt == ctx->options.end() || opt->second.compare("true") != 0)
            FAIL("dataset %s already exists", info.name.c_str());
        info.needs_overwriting = true;
        ctx->needs_overwriting = true;
    }
    ctx->udf_datasets.push_back(std::move(info));
    return true;
}

EXPORT bool libudf_compile(udf_context *ctx)
{
    CHECK_ARG_PTR(ctx);

    if (libudf_scan(ctx) == false)
        return false;

    std::vector<DatasetInfo> datasets(ctx->input_datasets);
    datasets.insert(datasets.end(), ctx->udf_datasets.begin(), ctx->udf_datasets.end());

    ctx->bytecode = ctx->backend->compile(
        ctx->udf_file,
        ctx->compound_declarations,
        ctx->sourcecode,
        datasets);
    if (ctx->bytecode.size() == 0)
        FAIL("failed to compile UDF file");

    return true;
}

EXPORT bool libudf_store(udf_context *ctx)
{
    CHECK_ARG_PTR(ctx);
    CHECK_ARG_PTR(ctx->backend);
    bool retval = true;

    // It is possible to write UDFs that define several datasets in a single
    // file. On the loop below we check which of the output datasets already
    // exist on the target file. Those that already exist are pushed into the
    // 'to_delete' vector so we can safely overwrite them.
    std::vector<std::string> to_delete;
    for (auto &info: ctx->udf_datasets)
        if (info.needs_overwriting)
            to_delete.push_back(info.name);

    HDF5_Handler h5(ctx->hdf5_file);
    if (h5.openFile(H5F_ACC_RDWR) == false)
        return false;
    if (h5.deleteDatasets(to_delete) == false)
        return false;

    for (auto &info: ctx->udf_datasets)
    {
        // Configure the HDF5-UDF filter
        if (h5.configureFilter(info.dimensions.size(), info.dimensions.data()) == false)
            return false;

        /// Prepare metadata (JSON payload)
        std::vector<std::string> input_dataset_names, scratch_dataset_names;
        std::transform(
            ctx->input_datasets.begin(),
            ctx->input_datasets.end(),
            std::back_inserter(input_dataset_names),
            [](DatasetInfo &info) -> std::string { return info.name; });

        for (auto &other: ctx->udf_datasets)
            if (other.name.compare(info.name) != 0)
                scratch_dataset_names.push_back(other.name);

        // Remove the output string size (if given) from the datatype
        auto payload_datatype = info.datatype;
        auto sep = payload_datatype.find("(");
        if (sep != std::string::npos)
            payload_datatype = payload_datatype.substr(0, sep);

        // Sign datasets and the UDF
        SignatureHandler signature;
        auto blob = signature.signPayload(
            (const uint8_t *) &ctx->bytecode[0], ctx->bytecode.length());
        if (blob == NULL)
            FAIL("failed to sign UDF");

        // JSON payload
        auto opt = ctx->options.find("save_sourcecode");
        bool save_sourcecode = opt != ctx->options.end() && opt->second.compare("true") == 0;

        json jas;
        jas["output_dataset"] = info.name;
        jas["output_resolution"] = info.dimensions;
        jas["output_datatype"] = payload_datatype;
        jas["input_datasets"] = input_dataset_names;
        jas["scratch_datasets"] = scratch_dataset_names;
        jas["bytecode_size"] = blob->size;
        jas["backend"] = ctx->backend->name();
        jas["source_code"] = save_sourcecode ? ctx->sourcecode : "";
        jas["api_version"] = 2;
        jas["signature"] = {
            {"public_key", blob->public_key_base64},
            {"name", blob->metadata["name"]},
            {"email", blob->metadata["email"]}
        };

        std::string jas_str = jas.dump();
        size_t payload_size = jas_str.length() + ctx->bytecode.size() + 1;
        INFO("\n%s dataset header:\n%s\n", info.name.c_str(), jas.dump(4, ' ', false, 45).c_str());
        if (ctx->compound_declarations.size())
            INFO("\nData structures available to the UDF:\n%s\n", ctx->compound_declarations.c_str());

        // Sanity check: the JSON and the bytecode must fit in the dataset
        hsize_t grid_size = std::accumulate(std::begin(info.dimensions), std::end(info.dimensions), 1, std::multiplies<hsize_t>());
        if (payload_size > (grid_size * H5Tget_size(info.hdf5_datatype)))
            FAIL("len(JSON+bytecode) > UDF dataset dimensions");

        // Prepare payload data
        char *payload = (char *) calloc(grid_size, H5Tget_size(info.hdf5_datatype));
        char *p = payload;
        memcpy(p, &jas_str[0], jas_str.length());
        p += jas_str.length();
        *p = '\0';
        memcpy(&p[1], blob->data, blob->size);

        // Create UDF dataset
        retval = h5.createUserDefinedDataset(info.name, info.hdf5_datatype, payload);

        // Close and release resources
        free(payload);
        delete blob;
    }

    return retval;
}


///////////////////////////////////////////////////////
// Private functions previously exposed as public API
///////////////////////////////////////////////////////

static bool libudf_scan(udf_context *ctx)
{
    CHECK_ARG_PTR(ctx);

    // Check that input files exist
    struct stat statbuf;
    std::vector<std::string> input_files = {ctx->hdf5_file, ctx->udf_file};
    for (auto input_file: input_files)
        if (stat(input_file.c_str(), &statbuf) < 0)
            FAIL("%s: %s", input_file.c_str(), strerror(errno));

    // Check that UDF file extension is recognized
    ctx->backend = getBackendByFileExtension(ctx->udf_file);
    if (! ctx->backend)
        FAIL("could not identify a parser for %s", ctx->udf_file.c_str());

    // Identify datasets that the UDF depends on by scanning
    // the UDF file and looking for explicit calls to lib.getData().
    // Note that any output (UDF) datasets will also be identified by
    // this scan. We tell which datasets are input and which are output
    // by checking which of the datasets already exist on the destination
    // file. Naturally, if the user is attempting to overwrite existing
    // UDF datasets then this scan alone will not be able to disambiguate.
    // In that case, the user needs to explicitly specify the UDF dataset 
    // names and their dimensions/datatypes.
    for (auto &name: ctx->backend->udfDatasetNames(ctx->udf_file))
    {
        HDF5_Handler h5(ctx->hdf5_file);
        if (h5.openFile(H5F_ACC_RDONLY) == false)
            return false;

        auto is_input_dataset = isInputDataset(name, ctx);
        if (h5.hasDataset(name) && is_input_dataset)
        {
            if (! h5.openDataset(name, false) || ! h5.extractInfo())
                return false;

            // DatasetInfo entry we'll want to append
            DatasetInfo info(name, h5.dimensions, h5.datatype_name, h5.hdf5_datatype);

            // Check that the datatype is supported by our implementation
            if (strcmp(h5.datatype_name, "compound") == 0)
            {
                info.members = getCompoundMembers(h5.hdf5_datatype);
                if (info.members.size() == 0)
                    FAIL("failed parsing dataset %s from %s", name.c_str(), ctx->hdf5_file.c_str());
                else if (ctx->compound_declarations.size())
                    ctx->compound_declarations += "\n";
                ctx->compound_declarations += ctx->backend->compoundToStruct(info, false);
            }
            else if (strcmp(h5.datatype_name, "string") == 0)
            {
                // Because HDF5 has both variable- and fixed-sized strings there
                // are two possible ways to define a string element: (1) with
                // 'char varname[size]' and (2) with 'char *varname'. Here we embed
                // string variables into packed structures so that UDFs can easily
                // iterate over them with for loops and basic pointer arithmetic.
                bool is_varstring = H5Tis_variable_str(h5.hdf5_datatype);
                size_t member_size = H5Tget_size(h5.hdf5_datatype);
                info.members.push_back(info.getStringDeclaration(is_varstring, member_size));
                if (ctx->compound_declarations.size())
                    ctx->compound_declarations += "\n";
                ctx->compound_declarations += ctx->backend->compoundToStruct(info, true);
            }

            info.printInfo("Input");
            ctx->input_datasets.push_back(std::move(info));
        }
        else if (! udfDatasetAlreadySeen(name, ctx))
        {
            // This is an output (UDF) dataset
            DatasetInfo info(name, std::vector<hsize_t>(), "", -1);
            info.is_input_dataset = false;
            ctx->udf_datasets.push_back(std::move(info));
        }
    }

    if (ctx->udf_datasets.size() == 0)
        FAIL("output dataset(s) already exist or couldn't be identified");

    return libudf_validate_udf_datasets(ctx);
}

static bool libudf_validate_udf_datasets(udf_context *ctx)
{
    CHECK_ARG_PTR(ctx);
    CHECK_ARG_PTR(ctx->backend);

    for (auto &info: ctx->udf_datasets)
    {
        if (info.datatype.compare(0, 6, "string") == 0)
        {
            // Ideally, we'd like to output string datatypes using variable size;
            // that would make our life much easier. However, using variable string
            // sizes means that H5Dwrite() needs to call strlen() on each member of
            // the write data, and that fails badly because we're using the write
            // data buffer to hold the UDF bytecode -- so that call to strlen() is
            // likely to crash the application.
            int string_size = getUserStringSize(info.datatype);
            if (string_size < 0)
                FAIL("failed to parse string size in '%s'", info.datatype.c_str());

            if (H5Tset_size(info.hdf5_datatype, string_size) < 0)
                FAIL("failed to set dataset %#lx to variable size",
                    static_cast<unsigned long>(info.hdf5_datatype));

            // Embed the output variable in a packed structure so that the
            // UDFs can easily iterate over its members with for loops and
            // basic pointer arithmetic.
            info.members.push_back(
                info.getStringDeclaration(false, string_size));
            if (ctx->compound_declarations.size())
                ctx->compound_declarations += "\n";
            ctx->compound_declarations += ctx->backend->compoundToStruct(info, true);
        }
        else if (info.datatype.compare("compound") == 0)
        {
            // Embed the output variable in a packed structure so that the
            // UDFs can easily iterate over its members with for loops and
            // basic pointer arithmetic.
            if (ctx->compound_declarations.size())
                ctx->compound_declarations += "\n";
            ctx->compound_declarations += ctx->backend->compoundToStruct(info, false);
        }

        if (info.dimensions.size() > 0) {
            info.printInfo("User-defined");
            continue;
        }

        // Make sure that we have at least one output dataset
        if (ctx->input_datasets.size() == 0)
            FAIL("cannot determine dims and type of UDF dataset %s. Please specify.",
                info.name.c_str());

        // Require that all input datasets have the same dimensions and type
        for (size_t i=1; i<ctx->input_datasets.size(); ++i)
        {
            auto &input = ctx->input_datasets;
            if (! H5Tequal(input[i].hdf5_datatype, input[i-1].hdf5_datatype))
                FAIL("can't determine type of UDF dataset %s. Please specify.", info.name.c_str());
            else if (input[i].dimensions != input[i-1].dimensions)
                FAIL("can't determine dimensions of UDF dataset %s. Please specify.", info.name.c_str());
        }

        // We're all set: copy attributes from the first input dataset
        info.hdf5_datatype = H5Tcopy(ctx->input_datasets[0].hdf5_datatype);
        info.datatype = ctx->input_datasets[0].datatype;
        info.dimensions = ctx->input_datasets[0].dimensions;
        info.printInfo("User-defined");
    }

    return true;
}