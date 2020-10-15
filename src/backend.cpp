/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: backend.cpp
 *
 * Interfaces with supported code parsers and generators.
 */
#include <algorithm>
#include <fstream>
#include "backend.h"
#ifdef ENABLE_CPP
#include "cpp_backend.h"
#endif
#ifdef ENABLE_LUA
#include "lua_backend.h"
#endif
#ifdef ENABLE_PYTHON
#include "python_backend.h"
#endif

std::string Backend::assembleUDF(
    std::string udf_file, std::string template_file, std::string placeholder, std::string extension)
{
    std::ifstream ifs(udf_file);
    if (! ifs.is_open())
    {
        fprintf(stderr, "Failed to open %s\n", udf_file.c_str());
        return "";
    }
    std::string inputFileBuffer(
		(std::istreambuf_iterator<char>(ifs)),
        (std::istreambuf_iterator<char>()));

    /* Basic check: does the template file exist? */
    if (template_file.size() == 0)
    {
        fprintf(stderr, "Failed to find UDF template file\n");
        return "";
    }
    std::ifstream ifstr(template_file);
    std::string udf(
		(std::istreambuf_iterator<char>(ifstr)),
        (std::istreambuf_iterator<char>()));

    /* Basic check: is the template string present in the template file? */
    auto start = udf.find(placeholder);
    if (start == std::string::npos)
    {
        fprintf(stderr, "Failed to find placeholder string in %s\n",
            template_file.c_str());
        return "";
    }

    /* Embed UDF string in the template */
    auto completeCode = udf.replace(start, placeholder.length(), inputFileBuffer);

    /* Compile the code */
    auto out_file = writeToDisk(completeCode.data(), completeCode.size(), extension);
    if (out_file.size() == 0)
    {
        fprintf(stderr, "Will not be able to compile the UDF code\n");
        return "";
    }

    return out_file;
}

std::string Backend::writeToDisk(const char *data, size_t size, std::string extension)
{
    char *tmp = getenv("TMPDIR") ? : (char *) "/tmp";
    char path[PATH_MAX];
    std::ofstream tmpfile;
    snprintf(path, sizeof(path)-1, "%s/hdf5-udf-XXXXXX%s", tmp, extension.c_str());
    if (mkstemps(path, extension.size()) < 0){
        fprintf(stderr, "Error creating temporary file.\n");
        return std::string("");
    }
    tmpfile.open(path);
    tmpfile.write(data, size);
    tmpfile.flush();
    tmpfile.close();
    return std::string(path);
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
    return NULL;
}
