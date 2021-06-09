/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: sandbox_macos.cpp
 *
 * High-level interfaces for system call interception on macOS.
 */
#include "sandbox_macos.h"
#include "json.hpp"

using json = nlohmann::json;

////////////////////////////
// Tracer (parent) process
////////////////////////////

bool MacOSSandbox::initParent(std::string libpath, const json &rules, pid_t tracee_pid)
{
    // TODO
    return true;
}

///////////////////////////
// Tracee (child) process
///////////////////////////

bool MacOSSandbox::initChild(std::string libpath, const json &rules)
{
    // TODO
    return true;
}
