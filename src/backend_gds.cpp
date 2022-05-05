/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: backend_gds.cpp
 *
 * NVIDIA GPUDirect Storage backend implementation.
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
#include <cuda_runtime.h>
#include <cuda.h>
#include "sharedlib_manager.h"
#include "udf_template_cu.h" // use the CUDA template file
#include "backend_cpp.h" // reuse methods from the C++ backend
#include "backend_gds.h"
#include "anon_mmap.h"
#include "dataset.h"
#include "os.h"

/* This backend's name */
std::string GDSBackend::name()
{
    return "NVIDIA-GDS";
}

/* Extension managed by this backend */
std::string GDSBackend::extension()
{
    return ".cu";
}

/* Compile CU to a shared object using NVCC. Returns the shared object as a string. */
std::string GDSBackend::compile(
    std::string udf_file,
    std::string compound_declarations,
    std::string &source_code,
    std::vector<DatasetInfo> &datasets)
{
    CppBackend cpp;
    std::string methods_decl, methods_impl;
    cpp.generateUDFMethods(datasets, methods_decl, methods_impl);

    AssembleData data = {
        .udf_file                 = udf_file,
        .template_string          = std::string((char *) udf_template_cu),
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

    // Build the UDF
    char *cmd[] = {
        (char *) "/usr/local/cuda/bin/nvcc",
        (char *) "-Xcompiler",
        (char *) "-fpic,-O3",
        (char *) "-shared",
        (char *) "-Xlinker",
        (char *) "-L/usr/local/cuda/targets/x86_64-linux/lib",
        (char *) "-lcufile",
        (char *) "-I/usr/local/cuda/targets/x86_64-linux/include",
        (char *) "-o",
        (char *) output.c_str(),
        (char *) cpp_file.c_str(),
        (char *) "-Xptxas",
        (char *) "-O3",
        (char *) "-gencode",
        (char *) "arch=compute_80,code=sm_80",
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

    return CppBackend::compressBuffer(bytecode.data(), bytecode.size());
}

/* Execute the user-defined-function embedded in the given buffer */
bool GDSBackend::run(
    const std::string libpath,
    const std::vector<DatasetInfo> &input_datasets,
    const DatasetInfo &output_dataset,
    const char *output_cast_datatype,
    const char *sharedlib_data,
    size_t sharedlib_data_size,
    const json &rules)
{
    /* Decompress the shared library */
    std::string decompressed_shlib = CppBackend::decompressBuffer(
        sharedlib_data, sharedlib_data_size);
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
     * Differently from the C++ backend, the GDS backend doesn't run the
     * UDF in a separate proces; Among the limitations of CUDA GDS, one
     * cannot call fork() after libcufile has been used.
     */
    SharedLibraryManager shlib;
    if (shlib.open(so_file) == false)
        return false;

    /* Get references to the UDF and the APIs defined in our C++ template */
    void (*udf)(void) = (void (*)()) shlib.loadsym("dynamic_dataset");
    auto hdf5_udf_data =
        static_cast<std::vector<void *>*>(shlib.loadsym("hdf5_udf_data"));
    auto hdf5_udf_names =
        static_cast<std::vector<const char *>*>(shlib.loadsym("hdf5_udf_names"));
    auto hdf5_udf_types =
        static_cast<std::vector<const char *>*>(shlib.loadsym("hdf5_udf_types"));
    auto hdf5_udf_dims =
        static_cast<std::vector<std::vector<hsize_t>>*>(shlib.loadsym("hdf5_udf_dims"));
    int (*hdf5_udf_last_cuda_error)(void) = (int (*)()) shlib.loadsym("hdf5_udf_last_cuda_error");
    if (! udf || ! hdf5_udf_data || ! hdf5_udf_names ||
        ! hdf5_udf_types || ! hdf5_udf_dims || ! hdf5_udf_last_cuda_error)
        return false;

    /*
     * Populate vector of dataset names, sizes, and types. Note that we have
     * to manually handle the reference counting of hdf5_datatype by
     * calling each member's reopenDatatype() method.
     */
    std::vector<DatasetInfo> dataset_info;
    dataset_info.push_back(output_dataset);
    dataset_info.insert(dataset_info.end(), input_datasets.begin(), input_datasets.end());

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

    /* Run the UDF */
    udf();

    cudaError_t err = static_cast<cudaError_t>(hdf5_udf_last_cuda_error());
    if (err != cudaSuccess)
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    else
        cudaDeviceSynchronize();

    /* Flush stdout buffer so we don't miss any messages echoed by the UDF */
    fflush(stdout);

    unlink(so_file.c_str());
    return err == cudaSuccess;
}

/* Scan the UDF file for references to HDF5 dataset names */
std::vector<std::string> GDSBackend::udfDatasetNames(std::string udf_file)
{
    std::vector<std::string> output;
    std::string input;

    // We already rely on NVCC to build the code, so just invoke its
    // preprocessor to get rid of comments and identify calls to our API
    char *cmd[] = {
        (char *) "/usr/local/cuda/bin/nvcc",
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
std::string GDSBackend::compoundToStruct(const DatasetInfo &info, bool hardcoded_name)
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

/* Allocate memory for an input or scratch dataset in device memory */
void *GDSBackend::alloc(size_t size)
{
    DeviceMemory *devicememory = new DeviceMemory(size);
    if (! devicememory->dev_mem)
    {
        delete devicememory;
        return NULL;
    }
    memory_map[devicememory->dev_mem] = devicememory;
    return devicememory->dev_mem;
}

/* Free memory previously allocated for an input or scratch dataset */
void GDSBackend::free(void *mem)
{
    std::map<void*, DeviceMemory*>::iterator it = memory_map.find(mem);
    if (it != memory_map.end())
    {
        DeviceMemory *devicememory = it->second;
        delete devicememory;
        memory_map.erase(it);
    }
}

/* Copy data from device memory to a newly allocated memory chunk in the host */
void *GDSBackend::deviceToHost(void *dev_mem, size_t size)
{
    std::map<void*, DeviceMemory*>::iterator it = memory_map.find(dev_mem);
    if (it == memory_map.end())
    {
        fprintf(stderr, "Provided device memory address not managed by ourselves\n");
        return NULL;
    }
    auto mm = it->second;
    if (size != mm->size)
    {
        fprintf(stderr, "Buffer has %ld bytes on device memory but %ld bytes on system memory\n",
            mm->size, size);
        return NULL;
    }

    void *host_mem = malloc(size);
    if (! host_mem)
    {
        fprintf(stderr, "Not enough memory allocating host memory for output grid\n");
        return NULL;
    }
    if (DirectDataset::copyToHost(*(it->second), &host_mem) == false)
    {
        fprintf(stderr, "Failed to copy output grid memory from device to host\n");
        free(host_mem);
        return NULL;
    }
    return host_mem;
}

/* Get a reference to the memory handler of the given device memory address */
DeviceMemory *GDSBackend::memoryHandler(void *dev_mem)
{
    std::map<void*, DeviceMemory*>::iterator it = memory_map.find(dev_mem);
    if (it == memory_map.end())
    {
        fprintf(stderr, "Provided device memory address not managed by ourselves\n");
        return NULL;
    }
    return it->second;
}

/* Zeroes out a range of memory previously allocated for an input or scratch dataset */
void GDSBackend::clear(void *dev_mem, size_t size)
{
    std::map<void*, DeviceMemory*>::iterator it = memory_map.find(dev_mem);
    if (it == memory_map.end())
    {
        fprintf(stderr, "Provided device memory address not managed by ourselves\n");
        return;
    }
    DeviceMemory *devicememory = it->second;
    if (devicememory->size != size)
    {
        fprintf(stderr, "Error: the API expects to clear out the entire memory region\n");
        return;
    }
    it->second->clear();
}