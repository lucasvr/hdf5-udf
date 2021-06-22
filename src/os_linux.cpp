/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: os_linux.cpp
 *
 * Linux-specific routines
 */
#ifdef __linux__
#include <sys/utsname.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <seccomp.h>
#include <unistd.h>
#include <dirent.h>
#include <string.h>
#include <pwd.h>
#include "os.h"
#ifdef ENABLE_SANDBOX
#include "sandbox_linux.h"
#endif

#ifndef SYS_SECCOMP
#define SYS_SECCOMP 1
#endif

std::vector<std::string> os::openedH5Files()
{
    const char *proc = "/proc/self/fd";
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
        {
            char target[PATH_MAX];
            memset(target, 0, sizeof(target));
            if (readlink(fname.c_str(), target, sizeof(target)-1) > 0)
                out.push_back(target);
        }
    }
    closedir(d);
    return out;
}

std::string os::sharedLibraryName(std::string name)
{
    return "lib" + name + ".so";
}

std::string os::configDirectory()
{
    const char *xdg_config_home = getenv("XDG_CONFIG_HOME");
    const char *home = getenv("HOME");
    if (xdg_config_home)
        return std::string(xdg_config_home) + "/hdf5-udf/";
    else if (home)
        return std::string(home) + "/.config/hdf5-udf/";
    else
    {
        fprintf(stderr, "Error: $HOME is not set, falling back to /tmp\n");
        auto pwp = getpwuid(getuid());
        if (pwp)
            return "/tmp/hdf5-udf." + std::string(pwp->pw_name) + "/";
        else
            return "/tmp/hdf5-udf/";
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
    int ret = seccomp_syscall_resolve_name(name.c_str());
    if (ret == __NR_SCMP_ERROR)
        return -1;
    return ret;
}

bool os::initChildSandbox(std::string libpath, const nlohmann::json &rules)
{
    return LinuxSandbox().initChild(libpath, rules);
}

bool os::initParentSandbox(std::string libpath, const nlohmann::json &rules, pid_t tracee_pid)
{
    return LinuxSandbox().initParent(libpath, rules, tracee_pid);
}
#endif // ENABLE_SANDBOX

#endif // __linux__
