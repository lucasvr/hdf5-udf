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
#include <vector>
#include <string>
#include "sharedlib_manager.h"
#include "json.hpp"

using json = nlohmann::json;

class Sandbox {
public:
    Sandbox() {}
    ~Sandbox() {}
    bool init(
        std::string filterpath,
        const std::vector<std::string> &paths_allowed,
        const json &rules);

private:
    bool initSeccomp(const json &rules);
};

#endif