/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: sandbox_macos.h
 *
 * Sandboxing routines to prevent certain system calls
 * from being executed by the user-defined-functions.
 */
#ifndef __sandbox_macos_h
#define __sandbox_macos_h

#include "sandbox.h"

class MacOSSandbox : public Sandbox {
public:
    MacOSSandbox() {}
    ~MacOSSandbox() {}

    // Entry point called by the child process (i.e., the one that runs the UDF).
    bool initChild(std::string libpath, const nlohmann::json &rules);

    // Entry point called by the parent process.
    bool initParent(std::string libpath, const nlohmann::json &rules, pid_t tracee_pid);
};

#endif
