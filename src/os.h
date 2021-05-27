/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: os.h
 *
 * Non-portable routines
 */
#ifndef __os_h
#define __os_h

#include <sys/types.h>
#include <vector>
#include <string>
#include "json.hpp"

namespace os {

    // Get a list of open files from /proc. This is a workaround for the
    // lack of an HDF5 Filter API to access the underlying file descriptor.
    std::vector<std::string> openedH5Files();

    // Convert a name into its os-specific shared library name.
    // Example: foo -> libfoo.so
    std::string sharedLibraryName(std::string name);

    // Default HDF5 plugin path
    std::string defaultPluginPath();

    // Config directory under the currently logged in user's home
    std::string configDirectory();

    // Convert a system call name to its number. Return -1 on error.
    int syscallNameToNumber(std::string name);

    // Configure the sandbox at the UDF (child) process
    bool initChildSandbox(std::string filterpath, const nlohmann::json &rules);

    // Configure the sandbox at the parent process
    bool initParentSandbox(std::string filterpath, const nlohmann::json &rules, pid_t tracee_pid);
}

#endif