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
#include <pcrecpp.h>
#include <sys/user.h>
#include <functional>
#include <algorithm>
#include <vector>
#include <deque>
#include <string>
#include "sharedlib_manager.h"
#include "json.hpp"

enum fs_access_mode {
    READONLY = 0,
    READWRITE = 1,
};

class Sandbox {
public:
    virtual ~Sandbox() {}

    // Entry point called by the child process (i.e., the one that runs the UDF).
    // This method loads Seccomp rules and configures the process so it waits
    // to be controlled by the parent with Ptrace.
    virtual bool initChild(std::string filterpath, const nlohmann::json &rules) {
        return true;
    }

    // Entry point called by the parent process. This method scans the config
    // file to retrieve the filesystem paths allowed to be accessed by the child
    // process (i.e., the UDF) and monitors the system calls it executes with
    // Ptrace. Filesystem path violations are punished with SIGKILL.
    virtual bool initParent(std::string filterpath, const nlohmann::json &rules, pid_t tracee_pid) {
        return true;
    }
};

#endif