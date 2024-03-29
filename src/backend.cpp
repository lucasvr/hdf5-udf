/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: backend.cpp
 *
 * Interfaces with supported code parsers and generators.
 */
#include <stdlib.h>
#include <algorithm>
#include <fstream>
#include "config.h"
#include "backend.h"
#ifdef ENABLE_CPP
#include "backend_cpp.h"
#endif
#ifdef ENABLE_LUA
#include "backend_lua.h"
#endif
#ifdef ENABLE_PYTHON
#include "backend_python.h"
#endif
#ifdef ENABLE_CUDA
#include "backend_cuda.h"
#endif
#include "os.h"

static std::string textReplace(
    std::string input,
    std::string placeholder,
    std::string newstr)
{
    if (newstr.size())
    {
        auto start = input.find(placeholder);
        if (start == std::string::npos)
            return "";
        return input.replace(start, placeholder.length(), newstr);
    }
    return input;
}

std::string Backend::assembleUDF(const AssembleData &data)
{
    std::ifstream ifs(data.udf_file);
    if (! ifs.is_open())
    {
        fprintf(stderr, "Failed to open %s\n", data.udf_file.c_str());
        return "";
    }
    std::string inputFileBuffer(
		(std::istreambuf_iterator<char>(ifs)),
        (std::istreambuf_iterator<char>()));

    std::string udf(data.template_string);

    /* Populate placeholders */
    udf = textReplace(udf, data.methods_decl_placeholder, data.methods_decl);
    if (udf.size() == 0)
    {
        fprintf(stderr, "Missing methods_decl placeholder in the template string\n");
        return "";
    }
    udf = textReplace(udf, data.methods_impl_placeholder, data.methods_impl);
    if (udf.size() == 0)
    {
        fprintf(stderr, "Missing methods_impl placeholder in the template string\n");
        return "";
    }
    udf = textReplace(udf, data.compound_placeholder, data.compound_decl);
    if (udf.size() == 0)
    {
        fprintf(stderr, "Missing compound placeholder in the template string\n");
        return "";
    }
    udf = textReplace(udf, data.callback_placeholder, inputFileBuffer);
    if (udf.size() == 0)
    {
        fprintf(stderr, "Missing callback placeholder in the template string\n");
        return "";
    }

    /* Write the final code to disk */
    auto out_file = writeToDisk(udf.data(), udf.size(), data.extension);
    if (out_file.size() == 0)
    {
        fprintf(stderr, "Will not be able to compile the UDF code\n");
        return "";
    }

    // Use forward-slash path separator
    std::replace(out_file.begin(), out_file.end(), '\\', '/');
    return out_file;
}

std::string Backend::writeToDisk(const char *data, size_t size, std::string extension)
{
    std::string path = os::makeTemporaryFile("hdf5-udf-XXXXXX", extension);
    if (path.size() == 0)
    {
        fprintf(stderr, "Error creating temporary file.\n");
        return std::string("");
    }
    std::ofstream tmpfile;
    tmpfile.open(path, std::ofstream::binary);
    tmpfile.write(data, size);
    tmpfile.flush();
    tmpfile.close();

    // Use forward-slash path separator
    std::replace(path.begin(), path.end(), '\\', '/');
    return std::string(path);
}

std::string Backend::sanitizedName(std::string name)
{
    // Truncate the string at any of the following tokens
    const char *truncate_at[] = {"(", "[", NULL};
    for (int i=0; truncate_at[i] != NULL; ++i)
        while (true)
        {
            auto index = name.find(truncate_at[i]);
            if (index == std::string::npos)
                break;
            name = name.substr(0, index);
        }

    // Replace the following tokens by an underscore
    const char *replace_at[] = {"-", " ", ".", NULL};
    for (int i=0; replace_at[i] != NULL; ++i)
        while (true)
        {
            auto index = name.find(replace_at[i]);
            if (index == std::string::npos)
                break;
            name.replace(index, 1, "_");
        }

    // Remove unwanted tokens from the end of the string
    for (ssize_t i=name.size()-1; i>=0; --i)
    {
        if (name[i] != '_')
            break;
        name = name.substr(0, i);
    }

    // Put string to lowercase
    std::transform(name.begin(), name.end(), name.begin(),
        [](unsigned char c){ return std::tolower(c); });


    return name;
}

// Get a backend by their name (e.g., "LuaJIT")
Backend *getBackendByName(std::string name)
{
#ifdef ENABLE_LUA
    if (name.compare("LuaJIT") == 0)
        return static_cast<Backend *>(new LuaBackend());
#endif
#ifdef ENABLE_PYTHON
    if (name.compare("CPython") == 0)
        return static_cast<Backend *>(new PythonBackend());
#endif
#ifdef ENABLE_CPP
    if (name.compare("C++") == 0)
        return static_cast<Backend *>(new CppBackend());
#endif
#ifdef ENABLE_CUDA
    if (name.compare("CUDA") == 0)
        return static_cast<Backend *>(new CudaBackend());
#endif
    return NULL;
}

// Get a backend by file extension (e.g., ".lua")
Backend *getBackendByFileExtension(std::string name)
{
    auto sep = name.rfind(".");
    if (sep == std::string::npos)
        return NULL;

    auto sameString = [](const std::string a, const std::string b)
    {
        return std::equal(
            a.begin(), a.end(),
            b.begin(), b.end(),
            [](char a, char b) { return tolower(a) == tolower(b); });
    };

    auto ext = name.substr(sep);
#ifdef ENABLE_LUA
    if (sameString(ext, ".lua"))
        return static_cast<Backend *>(new LuaBackend());
#endif
#ifdef ENABLE_PYTHON
    if (sameString(ext, ".py"))
        return static_cast<Backend *>(new PythonBackend());
#endif
#ifdef ENABLE_CPP
    if (sameString(ext, ".cpp"))
        return static_cast<Backend *>(new CppBackend());
#endif
#ifdef ENABLE_CUDA
    if (sameString(ext, ".cu"))
        return static_cast<Backend *>(new CudaBackend());
#endif
    return NULL;
}
