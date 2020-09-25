/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: sandbox.h
 *
 * Sandboxing routines to prevent certain system calls
 * from being executed by the user-defined-functions.
 */
#ifndef __sandbox_h
#define __sandbox_h

#include <stdbool.h>
#include <functional>
#include <algorithm>
#include <string>
#include "sharedlib_manager.h"

class Sandbox {
public:
    Sandbox() {}
    ~Sandbox() {}
    bool init(std::string filterpath);

private:
    std::string extractSymbol(std::string elf, std::string symbol_name);
    SharedLibraryManager shlib;
};

#endif