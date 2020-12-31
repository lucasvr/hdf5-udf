/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: cpp_backend.cpp
 *
 * C++ code parser and shared library generation/execution.
 */
#include <stdio.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include "sharedlib_manager.h"
#include "cpp_backend.h"
#include "anon_mmap.h"
#include "dataset.h"
#include "miniz.h"
#ifdef ENABLE_SANDBOX
#include "sandbox.h"
#endif

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

/* Compile C to a shared object using GCC. Returns the shared object as a string. */
std::string CppBackend::compile(
    std::string udf_file,
    std::string template_file,
    std::string compound_declarations,
    std::vector<DatasetInfo> &input_datasets)
{
    std::string methods_decl, methods_impl, spaces;
    for (auto &info: input_datasets)
    {
        if (info.datatype.compare("varstring") == 0)
        {
            auto name = sanitizedName(info.name);
            methods_decl += "const char *string(" + name + "_t &element);\n" + spaces;
            spaces = "    ";
            methods_impl += "const char *UserDefinedLibrary::string(" + name + "_t &element)\n{\n";
            methods_impl += spaces + "return static_cast<const char *>(element.value);\n";
            methods_impl += "}\n\n";
        }
    }
    AssembleData data = {
        .udf_file                 = udf_file,
        .template_file            = template_file,
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

    std::string output = udf_file + ".so";
    pid_t pid = fork();
    if (pid == 0)
    {
        // Child process. Note that, when building with Clang, we disable
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
#endif
            (char *) "-rdynamic",
            (char *) "-shared",
            (char *) "-fPIC",
            (char *) "-s",
            (char *) "-o",
            (char *) output.c_str(),
            (char *) cpp_file.c_str(),
            (char *) "-lm",
            NULL
        };
        execvp(cmd[0], cmd);
    }
    else if (pid > 0)
    {
        // Parent
        int exit_status;
        wait4(pid, &exit_status, 0, NULL);
        if (exit_status != 0)
        {
            fprintf(stderr, "Failed to compile the UDF\n");
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
            unlink(output.c_str());
        }
        unlink(cpp_file.c_str());

        // Compress the data
        return compressBuffer(bytecode.data(), bytecode.size());
    }
    fprintf(stderr, "Failed to execute the C++ compiler\n");
    return "";
}

/* Compress the shared library object and return the result as a string */
std::string CppBackend::compressBuffer(const char *data, size_t usize)
{
    uint64_t csize = mz_compressBound(usize);
    std::string compressed;
    compressed.resize(csize);

    auto status = mz_compress(
        (uint8_t *) compressed.data(), &csize,
        (const uint8_t *) data, usize);
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
        (uint8_t *) uncompressed.data(), &usize,
        (const uint8_t *) data, csize);
    if (status != Z_OK)
    {
        fprintf(stderr, "Failed to uncompress shared library object: %d\n", status);
        return "";
    }
    return uncompressed;
}

/* Execute the user-defined-function embedded in the given buffer */
bool CppBackend::run(
    const std::string filterpath,
    const std::vector<DatasetInfo> input_datasets,
    const DatasetInfo output_dataset,
    const char *output_cast_datatype,
    const char *sharedlib_data,
    size_t sharedlib_data_size)
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
    auto ext = this->extension();
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
    if (! mm.create())
    {
        unlink(so_file.c_str());
        return false;
    }

    /*
     * Execute the user-defined-function under a separate process so that
     * seccomp can kill it (if needed) without crashing the entire program
     */
    pid_t pid = fork();
    if (pid == 0)
    {
        SharedLibraryManager shlib;
        if (shlib.open(so_file) == false)
            return false;

        /* Get references to the UDF and the APIs defined in our C++ template file */
        void (*udf)(void) = (void (*)()) shlib.loadsym("dynamic_dataset");
        auto hdf5_udf_data =
            static_cast<std::vector<void *>*>(shlib.loadsym("hdf5_udf_data"));
        auto hdf5_udf_names =
            static_cast<std::vector<const char *>*>(shlib.loadsym("hdf5_udf_names"));
        auto hdf5_udf_types =
            static_cast<std::vector<const char *>*>(shlib.loadsym("hdf5_udf_types"));
        auto hdf5_udf_dims =
            static_cast<std::vector<std::vector<hsize_t>>*>(shlib.loadsym("hdf5_udf_dims"));
        if (! udf || ! hdf5_udf_data || ! hdf5_udf_names || ! hdf5_udf_types || ! hdf5_udf_dims)
            return false;

        /* Let output_dataset.data point to the shared memory segment */
        DatasetInfo output_dataset_copy = output_dataset;
        output_dataset_copy.data = mm.mm;

        /* Populate vector of dataset names, sizes, and types */
        std::vector<DatasetInfo> dataset_info;
        dataset_info.push_back(output_dataset_copy);
        dataset_info.insert(
            dataset_info.end(), input_datasets.begin(), input_datasets.end());

        for (size_t i=0; i<dataset_info.size(); ++i)
        {
            hdf5_udf_data->push_back(dataset_info[i].data);
            hdf5_udf_names->push_back(dataset_info[i].name.c_str());
            hdf5_udf_types->push_back(dataset_info[i].getDatatype());
            hdf5_udf_dims->push_back(dataset_info[i].dimensions);
        }

        /* Prepare the sandbox if needed and run the UDF */
        bool ready = true;
#ifdef ENABLE_SANDBOX
        Sandbox sandbox;
        ready = sandbox.init(filterpath, std::vector<std::string>());
#endif
        if (ready)
        {
            // Run the UDF
            udf();

            // Flush stdout buffer so we don't miss any messages echoed by the UDF
            fflush(stdout);
        }

        /* Exit the process without invoking any callbacks registered with atexit() */
        _exit(ready ? 0 : 1);
    }
    else if (pid > 0)
    {
        /* Update output HDF5 dataset with data from shared memory segment */
        waitpid(pid, NULL, 0);
        memcpy(output_dataset.data, mm.mm, room_size);
    }

    unlink(so_file.c_str());
    return true;   
}

/* Scan the UDF file for references to HDF5 dataset names */
std::vector<std::string> CppBackend::udfDatasetNames(std::string udf_file)
{
    std::vector<std::string> output;

    // We already rely on GCC to build the code, so just invoke its
    // preprocessor to get rid of comments and identify calls to our API
    int pipefd[2];
    if (pipe(pipefd) < 0)
    {
        fprintf(stderr, "Failed to create pipe\n");
        return output;
    }

    pid_t pid = fork();
    if (pid == 0)
    {
        // Child: runs proprocessor, outputs to pipe
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[0]);
        close(pipefd[1]);
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
        execvp(cmd[0], cmd);
    }
    else if (pid > 0)
    {
        // Parent: reads from pipe, concatenating to 'input' string
        std::string input;
        fcntl(pipefd[0], F_SETFL, O_NONBLOCK);
        while (true)
        {
            char buf[8192];
            ssize_t n = read(pipefd[0], buf, sizeof(buf));
            if (n < 0 && errno == EWOULDBLOCK)
            {
                int exit_status = 0;
                if (waitpid(pid, &exit_status, WNOHANG) == pid)
                {
                    if (exit_status != 0)
                    {
                        fprintf(stderr, "Failed to run the C++ preprocessor\n");
                        return output;
                    }
                    break;
                }
                continue;
            }
            else if (n <= 0)
                break;
            input.append(buf, n);
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

        close(pipefd[0]);
        close(pipefd[1]);
    }
    return output;
}

/* Create a textual declaration of a struct given a compound map */
std::string CppBackend::compoundToStruct(const DatasetInfo info, bool hardcoded_name)
{
    // We use GCC's __attribute__((packed)) to ensure the structure
    // is byte-aligned. This is required so that we can iterate over
    // the data retrieved by H5Dread() with just a struct pointer.
    std::string cstruct = "struct __attribute__((packed)) " + sanitizedName(info.name) + "_t {\n";
    size_t current_offset = 0, pad = 0;
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