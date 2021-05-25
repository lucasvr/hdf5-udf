/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: os_linux.cpp
 *
 * Linux-specific routines
 */
#ifdef __linux__
#include <sys/types.h>
#include <sys/stat.h>
#include <seccomp.h>
#include <unistd.h>
#include <dirent.h>
#include <string.h>
#include <pwd.h>
#include "os.h"

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

std::string os::defaultPluginPath()
{
    return "/usr/local/hdf5/lib/plugin/";
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
        auto pwp = getpwuid(getuid());
        if (pwp)
            return "/tmp/hdf5-udf." + std::string(pwp->pw_name) + "/";
        else
            return "/tmp/hdf5-udf/";
    }
}

#endif // __linux__