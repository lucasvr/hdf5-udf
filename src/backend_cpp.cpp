/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: backend_cpp.cpp
 *
 * C++ code parser and shared library generation/execution.
 */
#include <stdio.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include "config.h"
#include "sharedlib_manager.h"
#include "udf_template_cpp.h"
#include "backend_cpp.h"
#include "anon_mmap.h"
#include "dataset.h"
#include "miniz.h"
#include "os.h"

/* This backend's name */
std::string CppBackend::name()
{
    return "C++";
}

/* Extension managed by this backend */
std::string CppBackend::extension()
{
    return ".cpp";
}

/* Generate UDF methods for string-based dataset */
void CppBackend::generateUDFMethods(
    const std::vector<DatasetInfo> &datasets,
    std::string &methods_decl,
    std::string &methods_impl)
{
    std::string spaces;
    for (auto &info: datasets)
    {
        if (info.datatype.compare(0, 6, "string") == 0)
        {
            auto name = sanitizedName(info.name);
            methods_decl += "const char *string(" + name + "_t &element);\n" + spaces;
            spaces = "    ";
            methods_impl += "const char *UserDefinedLibrary::string(" + name + "_t &element)\n{\n";
            methods_impl += spaces + "return static_cast<const char *>(element.value);\n";
            methods_impl += "}\n\n";

            methods_decl += "void setString(" + name + "_t &element, const char *format, ...);\n" + spaces;
            methods_impl += "void UserDefinedLibrary::setString(" + name + "_t &element, const char *format, ...)\n{\n";
            methods_impl += spaces + "va_list argptr;\n";
            methods_impl += spaces + "va_start(argptr, format);\n";
            methods_impl += spaces + "vsnprintf((char *) element.value, sizeof(element.value), format, argptr);\n";
            methods_impl += spaces + "va_end(argptr);\n";
            methods_impl += "}\n\n";
        }
    }
}

/* Compile C to a shared object using GCC. Returns the shared object as a string. */
std::string CppBackend::compile(
    std::string udf_file,
    std::string compound_declarations,
    std::string &source_code,
    std::vector<DatasetInfo> &datasets)
{
    std::string methods_decl, methods_impl;
    generateUDFMethods(datasets, methods_decl, methods_impl);

    AssembleData data = {
        .udf_file                 = udf_file,
        .template_string          = std::string((char *) udf_template_cpp),
        .compound_placeholder     = "// compound_declarations_placeholder",
        .compound_decl            = compound_declarations,
        .methods_decl_placeholder = "// methods_declaration_placeholder",
        .methods_decl             = methods_decl,
        .methods_impl_placeholder = "// methods_implementation_placeholder",
        .methods_impl             = methods_impl,
        .callback_placeholder     = "// user_callback_placeholder",
        .extension                = this->extension()
    };
    auto cpp_file = Backend::assembleUDF(data);
    if (cpp_file.size() == 0)
    {
        fprintf(stderr, "Will not be able to compile the UDF code\n");
        return "";
    }

    std::string sharedlib = os::sharedLibraryName("dummy");
    std::string ext = sharedlib.substr(sharedlib.find_last_of("."));
    std::string output = udf_file + ext;

    // Build the UDF. Note that, when building with Clang, we disable
    // -flto. That's because Travis CI, at the very least, ships without
    // Clang's Gold Linker, leading to compile-time failures. We're taking
    // a more safe path here to avoid having to workaround Travis' limitations.
    char *cmd[] = {
#ifdef __clang__
        (char *) "clang++",
        (char *) "-O3",
#else
        (char *) "g++",
        (char *) "-flto",
        (char *) "-Os",
        (char *) "-C",
        (char *) "-s",
#endif
#ifndef __MINGW64__
        (char *) "-rdynamic",
        (char *) "-fPIC",
#endif
        (char *) "-shared",
        (char *) "-std=c++14",
        (char *) "-o",
        (char *) output.c_str(),
        (char *) cpp_file.c_str(),
        (char *) "-lm",
        NULL
    };
    if (os::execCommand(cmd[0], cmd, NULL) == false)
    {
        fprintf(stderr, "Failed to build UDF\n");
        unlink(cpp_file.c_str());
        return "";
    }

    // Read generated shared library
    struct stat statbuf;
    std::string bytecode;
    if (stat(output.c_str(), &statbuf) == 0) {
        std::ifstream data(output, std::ifstream::binary);
        std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(data), {});
        bytecode.assign(buffer.begin(), buffer.end());
        data.close();
        unlink(output.c_str());
    }

    // Read source file
    std::ifstream ifs(cpp_file.c_str());
    source_code = std::string((std::istreambuf_iterator<char>(ifs)),
            (std::istreambuf_iterator<char>()));
    ifs.close();
    unlink(cpp_file.c_str());

    // Compress the data
    return compressBuffer(bytecode.data(), bytecode.size());
}

/* Compress the shared library object and return the result as a string */
std::string CppBackend::compressBuffer(const char *data, size_t usize)
{
    uint64_t csize = mz_compressBound(usize);
    std::string compressed;
    compressed.resize(csize);

    auto status = mz_compress(
        (uint8_t *) compressed.data(),
        (mz_ulong *) &csize,
        (const uint8_t *) data,
        usize);
    if (status != Z_OK)
    {
        fprintf(stderr, "Failed to compress input buffer\n");
        return "";
    }
    memcpy(&compressed[csize], &usize, sizeof(uint64_t));
    compressed.resize(csize + sizeof(uint64_t));
    return compressed;
}

/* Decompress the shared library object and return the result as a string */
std::string CppBackend::decompressBuffer(const char *data, size_t csize)
{
    /* Get original file size */
    uint64_t usize;
    memcpy(&usize, &data[csize-sizeof(uint64_t)], sizeof(uint64_t));
    csize -= sizeof(uint64_t);

    std::string uncompressed;
    uncompressed.resize(usize);

    auto status = mz_uncompress(
        (uint8_t *) uncompressed.data(),
        (mz_ulong *) &usize,
        (const uint8_t *) data,
        csize);
    if (status != Z_OK)
    {
        fprintf(stderr, "Failed to uncompress shared library object: %d\n", status);
        return "";
    }
    return uncompressed;
}

/* Execute the user-defined-function embedded in the given buffer */
bool CppBackend::run(
    const std::string libpath,
    const std::vector<DatasetInfo> &input_datasets,
    const DatasetInfo &output_dataset,
    const char *output_cast_datatype,
    const char *sharedlib_data,
    size_t sharedlib_data_size,
    const json &rules)
{
    /* Decompress the shared library */
    std::string decompressed_shlib = decompressBuffer(sharedlib_data, sharedlib_data_size);
    if (decompressed_shlib.size() == 0)
    {
        fprintf(stderr, "Will not be able to load the UDF function\n");
        return false;
    }

    /*
     * Unfortunately we have to make a trip to disk so we can dlopen()
     * and dlsym() the function we are looking for in a portable way.
     */
    std::string sharedlib = os::sharedLibraryName("dummy");
    std::string ext = sharedlib.substr(sharedlib.find_last_of("."));
    auto so_file = Backend::writeToDisk(decompressed_shlib.data(), decompressed_shlib.size(), ext);
    if (so_file.size() == 0)
    {
        fprintf(stderr, "Will not be able to load the UDF function\n");
        return false;
    }
    chmod(so_file.c_str(), 0755);

    /*
     * We want to make the output dataset writeable by the UDF. Because
     * the UDF is run under a separate process we have to use a shared
     * memory segment which both processes can read and write to.
     */
    size_t room_size = output_dataset.getGridSize() * output_dataset.getStorageSize();
    AnonymousMemoryMap mm(room_size);
    if (! mm.createMapFor(output_dataset.data))
    {
        unlink(so_file.c_str());
        return false;
    }

    // Execute the user-defined-function under a separate process so that
    // seccomp can kill it (if needed) without crashing the entire program
    //
    // Support for Windows is still experimental; there is no sandboxing as of
    // yet, and the OS doesn't provide a fork()-like API with similar semantics.
    // In that case we just let the UDF run in the same process space as the parent.
    // Note that we define fork() as a no-op that returns 0 so we can reduce the
    // amount of #ifdef blocks in the body of this function.
    bool retval = false;
    pid_t pid = fork();
    if (pid == 0)
    {
        SharedLibraryManager shlib;
        if (shlib.open(so_file) == false)
            return false;

        /* Get references to the UDF and the APIs defined in our C++ template */
        void (*udf)(void) = (void (*)()) shlib.loadsym("dynamic_dataset");
        auto hdf5_file_path = static_cast<const char **>(shlib.loadsym("hdf5_file_path"));
        auto hdf5_udf_data =
            static_cast<std::vector<void *>*>(shlib.loadsym("hdf5_udf_data"));
        auto hdf5_udf_names =
            static_cast<std::vector<const char *>*>(shlib.loadsym("hdf5_udf_names"));
        auto hdf5_udf_types =
            static_cast<std::vector<const char *>*>(shlib.loadsym("hdf5_udf_types"));
        auto hdf5_udf_dims =
            static_cast<std::vector<std::vector<hsize_t>>*>(shlib.loadsym("hdf5_udf_dims"));
        if (! udf || ! hdf5_file_path || ! hdf5_udf_data || ! hdf5_udf_names || ! hdf5_udf_types || ! hdf5_udf_dims)
            return false;

        /* Let output_dataset.data point to the shared memory segment */
        DatasetInfo output_dataset_copy(output_dataset);
        output_dataset_copy.data = mm.mm;

        /*
         * Populate vector of dataset names, sizes, and types. Note that we have
         * to manually account for proper reference counting of hdf5_datatype by
         * calling each member's reopenDatatype() method.
         */
        std::vector<DatasetInfo> dataset_info;
        dataset_info.push_back(output_dataset_copy);
        dataset_info.insert(
            dataset_info.end(), input_datasets.begin(), input_datasets.end());

        for (size_t i=0; i<dataset_info.size(); ++i)
        {
            /*
             * reopen the datatype to prevent the destructor of the objects
             * pushed into the vector from closing the handles borrowed from
             * output_dataset and input_datasets.
             */
            dataset_info[i].reopenDatatype();
            hdf5_udf_data->push_back(dataset_info[i].data);
            hdf5_udf_names->push_back(dataset_info[i].name.c_str());
            hdf5_udf_types->push_back(dataset_info[i].getDatatypeName());
            hdf5_udf_dims->push_back(dataset_info[i].dimensions);
        }

        *hdf5_file_path = this->hdf5_file_path.c_str();

        /* Prepare the sandbox if needed and run the UDF */
        bool ready = true;
#ifdef ENABLE_SANDBOX
        if (rules.contains("sandbox") && rules["sandbox"].get<bool>() == true)
            ready = os::initChildSandbox(libpath, rules);
#endif
        if (ready)
        {
            // Run the UDF
            udf();

            // Flush stdout buffer so we don't miss any messages echoed by the UDF
            fflush(stdout);
        }

        // Exit the process without invoking any callbacks registered with atexit()
        if (os::isWindows()) { retval = ready; } else { _exit(ready ? 0  : 1); }
    }
    else if (pid > 0)
    {
        bool need_waitpid = true;
#ifdef ENABLE_SANDBOX
        if (rules.contains("sandbox") && rules["sandbox"].get<bool>() == true)
        {
            retval = os::initParentSandbox(libpath, rules, pid);
            need_waitpid = false;
        }
#endif
        if (need_waitpid)
        {
            int status;
            waitpid(pid, &status, 0);
            retval = WIFEXITED(status) ? WEXITSTATUS(status) == 0 : false;
        }

        /* Update output HDF5 dataset with data from shared memory segment */
        memcpy(output_dataset.data, mm.mm, room_size);
    }

    unlink(so_file.c_str());
    return retval;
}

/* Scan the UDF file for references to HDF5 dataset names */
std::vector<std::string> CppBackend::udfDatasetNames(std::string udf_file)
{
    std::vector<std::string> output;
    std::string input;

    // We already rely on GCC to build the code, so just invoke its
    // preprocessor to get rid of comments and identify calls to our API
    char *cmd[] = {
#ifdef __clang__
        (char *) "clang++",
#else
        (char *) "g++",
        (char *) "-fpreprocessed",
        (char *) "-dD",
#endif
        (char *) "-E",
        (char *) udf_file.c_str(),
        (char *) NULL
    };
    if (os::execCommand(cmd[0], cmd, &input) == false)
    {
        fprintf(stderr, "Failed to run the C++ preprocessor\n");
        return output;
    }

    // Go through the output of the preprocessor one line at a time
    std::string line;
    std::istringstream iss(input);
    while (std::getline(iss, line))
    {
        size_t n = line.find("lib.getData");
        if (n != std::string::npos)
        {
            auto start = line.substr(n).find_first_of("\"");
            auto end = line.substr(n+start+1).find_first_of("\"");
            auto name = line.substr(n).substr(start+1, end);
            output.push_back(name);
        }
    }
    return output;
}

/* Create a textual declaration of a struct given a compound map */
std::string CppBackend::compoundToStruct(const DatasetInfo &info, bool hardcoded_name)
{
    // We use GCC's __attribute__((packed)) to ensure the structure
    // is byte-aligned. This is required so that we can iterate over
    // the data retrieved by H5Dread() with just a struct pointer.
    std::string cstruct = "struct __attribute__((packed)) " + sanitizedName(info.name) + "_t {\n";
    ssize_t current_offset = 0, pad = 0;
    for (auto &member: info.members)
    {
        if (member.offset > current_offset)
        {
            auto size = member.offset - current_offset;
            cstruct += "  char _pad" + std::to_string(pad++) +"["+ std::to_string(size) +"];\n";
        }
        current_offset = member.offset + member.size;
        cstruct += "  " + member.type + " " + (hardcoded_name ? "value" : sanitizedName(member.name));
        if (member.is_char_array)
            cstruct += "[" + std::to_string(member.size) + "]";
        cstruct += ";\n";
    }
    cstruct += "};\n";
    return cstruct;
}
