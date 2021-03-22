/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: main.cpp
 *
 * Compiles the UDF into executable form and embeds it as a
 * HDF5 dataset.
 */
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <map>
#include "hdf5-udf.h"

#define CHECK(ops) do { \
    if (! (ops)) \
        exit(1); \
} while(0)

int usage(int errcode, std::string unrecognized="")
{
    if (unrecognized.size())
        fprintf(stderr, "Error: unrecognized option '%s'\n\n", unrecognized.c_str());

    fprintf(stdout,
        "Syntax: hdf5-udf [flags] <hdf5_file> <udf_file> [udf_dataset..]\n\n"
        "Flags:\n"
        "  --overwrite              Overwrite existing UDF dataset(s)\n"
        "  --append-sourcecode      Include source code as metadata of the UDF\n\n"
        "Options:\n"
        "  hdf5_file                Input/output HDF5 file\n"
        "  udf_file                 File implementing the user-defined-function\n"
        "  udf_dataset              UDF dataset(s) to create. See syntax below.\n"
        "                           If omitted, dataset names are picked from udf_file\n"
        "                           and their resolutions/types are set to match the input\n"
        "                           datasets declared on that same file\n\n"
        "Syntax of 'udf_dataset' option for native data types:\n"
        "  name:resolution:type     name:       name of the UDF dataset\n"
        "                           resolution: XRES, XRESxYRES, or XRESxYRESxZRES\n"
        "                           type:       [u]int8, [u]int16, [u]int32, [u]int64, float,\n"
        "                                       double, string, or string(NN)\n"
        "                           If unset, strings have a fixed size of 32 characters.\n\n"
        "Syntax of 'udf_dataset' for compound data types:\n"
        "  name:{member:type[,member:type...]}:resolution\n\n"
        "Examples:\n"
        "hdf5-udf sample.h5 simple_vector.lua Simple:500:float\n"
        "hdf5-udf sample.h5 sine_wave.lua SineWave:100x10:int32\n"
        "hdf5-udf sample.h5 string_generator.lua 'Words:1000:string(80)'\n"
        "hdf5-udf sample.h5 udf.py /Group/Name/UserDefinedDataset:100x100:uint8\n"
        "hdf5-udf sample.h5 compound.cpp 'Observations:{id:uint8,location:string,temperature:float}:1000'\n\n");
    return errcode;
}

int main(int argc, char **argv)
{
    std::map<std::string, std::string> options;
    std::vector<std::string> args;

    // Parse command line options
    for (int i=0; i<argc; ++i)
    {
        if (strcmp(argv[i], "--overwrite") == 0)
            options["overwrite"] = "true";
        else if (strcmp(argv[i], "--append-sourcecode") == 0)
            options["save_sourcecode"] = "true";
        else if (strcmp(argv[i], "--help") == 0)
            exit(usage(0));
        else if (strncmp(argv[i], "--", 2) == 0)
            exit(usage(1, argv[1]));
        else
            args.push_back(argv[i]);
    }
    if (args.size() < 3)
        exit(usage(1));

    // Erase argv[0] from the args vector
    args.erase(args.begin());

    // Parse HDF5 + UDF file names and UDF dataset description
    std::string hdf5_file, udf_file;
    std::vector<std::string> descriptions;
    for (auto &arg: args)
    {
        if (hdf5_file.size() == 0)
            hdf5_file = arg;
        else if (udf_file.size() == 0)
            udf_file = arg;
        else
            descriptions.push_back(arg);
    }
    CHECK(hdf5_file.size() && udf_file.size());

    // Initialize the UDF library
    udf_context *ctx;
    CHECK(ctx = libudf_init(hdf5_file.c_str(), udf_file.c_str()));

    // Propagate key/value options to the library
    for (auto &option: options)
        CHECK(libudf_set_option(option.first.c_str(), option.second.c_str(), ctx));

    // Provide the description of all UDF datasets to the library
    for (auto &description: descriptions)
        CHECK(libudf_push_dataset(description.c_str(), ctx));

    // Compile and store the UDF on the target HDF5 file
    CHECK(libudf_compile(ctx));
    CHECK(libudf_store(ctx));

    // Done
    libudf_destroy(ctx);
    return 0;
}
