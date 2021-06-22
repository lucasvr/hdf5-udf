/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: os_macos.cpp
 *
 * macOS-specific routines
 */
#ifdef __APPLE__
#include <sys/utsname.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <unistd.h>
#include <dirent.h>
#include <string.h>
#include <pwd.h>
#include "os.h"
#ifdef ENABLE_SANDBOX
#include "sandbox_macos.h"
#endif

std::vector<std::string> os::openedH5Files()
{
    const char *proc = "/dev/fd";
    std::vector<std::string> out;
    DIR *d = opendir(proc);
    if (!d)
        return out;

    struct dirent *e;
    while ((e = readdir(d)) != NULL)
    {
        struct stat s;
        auto fname = std::string(proc) + "/" + std::string(e->d_name);
        if (stat(fname.c_str(), &s) == 0 && S_ISREG(s.st_mode))
            out.push_back(fname);
    }
    closedir(d);
    return out;
}

std::string os::sharedLibraryName(std::string name)
{
    return "lib" + name + ".dylib";
}

std::string os::configDirectory()
{
    const char *home = getenv("HOME");
    if (home)
        return std::string(home) + "/Library/HDF5-UDF/";
    else
    {
        fprintf(stderr, "Error: $HOME is not set, falling back to /tmp\n");
        auto pwp = getpwuid(getuid());
        if (pwp)
            return "/tmp/hdf5-udf." + std::string(pwp->pw_name) + "/";
        else
            return "/tmp/hdf5-udf";
    }
}

// os::makeTemporaryFile() implemented on os_posix.cpp
// os::setEnvironmentVariable() implemented on os_posix.cpp
// os::clearEnvironmentVariable() implemented on os_posix.cpp
// os::getUserInformation() implemented on os_posix.cpp
// os::createDirectory() implemented on os_posix.cpp
// os::execCommand() implemented on os_posix.cpp
// os::isWindows() implemented on os_posix.cpp

#ifdef ENABLE_SANDBOX
int os::syscallNameToNumber(std::string name)
{
    // TODO
    return 0;
}

bool os::initChildSandbox(std::string libpath, const nlohmann::json &rules)
{
    return MacOSSandbox().initChild(libpath, rules);
}

bool os::initParentSandbox(std::string libpath, const nlohmann::json &rules, pid_t tracee_pid)
{
    return MacOSSandbox().initParent(libpath, rules, tracee_pid);
}
#endif // ENABLE_SANDBOX

#endif // __APPLE__
