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

    /* Basic check: does the template file exist? */
    if (data.template_file.size() == 0)
    {
        fprintf(stderr, "Failed to find UDF template file\n");
        return "";
    }
    std::ifstream ifstr(data.template_file);
    std::string udf(
		(std::istreambuf_iterator<char>(ifstr)),
        (std::istreambuf_iterator<char>()));

    /* Check if the compound declaration placeholder present in the template file */
    auto compound_start = udf.find(data.compound_placeholder);
    if (compound_start == std::string::npos)
    {
        fprintf(stderr, "Failed to find compound placeholder string in %s\n",
            data.template_file.c_str());
        return "";
    }

    /* Replace compound placeholder with actual declaration string */
    if (data.compound_declarations.size()) {
        udf = udf.replace(
            compound_start,
            data.compound_placeholder.length(),
            data.compound_declarations);
    }

    /* Check if the template string is present in the template file */
    auto callback_start = udf.find(data.callback_placeholder);
    if (callback_start == std::string::npos)
    {
        fprintf(stderr, "Failed to find callback placeholder string in %s\n",
            data.template_file.c_str());
        return "";
    }

    /* Embed UDF string in the template */
    auto completeCode = udf.replace(
        callback_start,
        data.callback_placeholder.length(),
        inputFileBuffer);

    /* Write the final code to disk */
    auto out_file = writeToDisk(completeCode.data(), completeCode.size(), data.extension);
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
    const char *replace_at[] = {"-", " ", NULL};
    for (int i=0; replace_at[i] != NULL; ++i)
        while (true)
        {
            auto index = name.find(replace_at[i]);
            if (index == std::string::npos)
                break;
            name.replace(index, 1, "_");
        }

    // Remove unwanted tokens from the end of the string
    for (size_t i=name.size()-1; i>=0; --i)
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
