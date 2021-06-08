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

    // Create a temporary file according to the given template and extension
    std::string makeTemporaryFile(std::string template_name, std::string extension);

    // Define an environment variable, setenv() style
    bool setEnvironmentVariable(std::string name, std::string value);

    // Clear an environment variable, unsetenv() style
    bool clearEnvironmentVariable(std::string name);

    // Get user name, login, and hostname information
    bool getUserInformation(std::string &name, std::string &login, std::string &host);

    // Create a directory
    bool createDirectory(std::string name, int mode);

    // Execute a command
    bool execCommand(char *program, char *args[]);

    // Convert a system call name to its number. Return -1 on error.
    int syscallNameToNumber(std::string name);

    // Configure the sandbox at the UDF (child) process
    bool initChildSandbox(std::string filterpath, const nlohmann::json &rules);

    // Configure the sandbox at the parent process
    bool initParentSandbox(std::string filterpath, const nlohmann::json &rules, pid_t tracee_pid);
}

#endif
