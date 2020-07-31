/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: backend.cpp
 *
 * Interfaces with supported code parsers and generators.
 */
#include <algorithm>
#include "backend.h"
#include "cpp_backend.h"
#include "lua_backend.h"

// Get a backend by their name (e.g., "LuaJIT")
Backend *getBackendByName(std::string name)
{
    if (name.compare("LuaJIT") == 0)
        return static_cast<Backend *>(new LuaBackend());
    else if (name.compare("C++") == 0)
        return static_cast<Backend *>(new CppBackend());
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
    if (sameString(ext, ".lua"))
        return static_cast<Backend *>(new LuaBackend());
    else if (sameString(ext, ".cpp"))
        return static_cast<Backend *>(new CppBackend());
    return NULL;
}