/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: os_windows.cpp
 *
 * Windows-specific routines
 */
#ifdef __MINGW64__
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <string.h>
#include <direct.h>
#include <fcntl.h>
#include <errno.h>
#include <share.h>
#include <io.h>
#include "os.h"

std::vector<std::string> os::openedH5Files()
{
    // TODO
    std::vector<std::string> out;
    return out;
}

std::string os::sharedLibraryName(std::string name)
{
    return "lib" + name + ".dll";
}

std::string os::defaultPluginPath()
{
    const char *profile_dir = getenv("ALLUSERSPROFILE");
    if (profile_dir)
        return std::string(profile_dir) + "/hdf5/lib/plugin";
    fprintf(stderr, "Error: %%ALLUSERSPROFILE%% is not set\n");
    return "";
}

std::string os::configDirectory()
{
    const char *home = getenv("USERPROFILE");
    if (home)
        return std::string(home) + "/AppData/Local/HDF5-UDF/";
    fprintf(stderr, "Error: %%USERPROFILE%% is not set, falling back to %%PWD%%\n");
    return ".hdf5-udf/";
}

std::string os::makeTemporaryFile(std::string template_name, std::string extension)
{
    char fname[template_name.size()+1];
    sprintf(fname, "%s", template_name.c_str());
    char *xxxxxx = strstr(fname, "XXXXXX");
    if (! xxxxxx)
    {
        fprintf(stderr, "Error: malformed string provided to makeTemporaryFile()\n");
        return "";
    }
    const char *letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    for (int i=0; i<0x0fffffff; ++i)
    {
        for (int j=0; j<6; ++j)
            xxxxxx[j] = letters[rand() % strlen(letters)];
        int fd = _sopen(fname, _O_RDWR | _O_CREAT | _O_EXCL, _SH_DENYRW);
        if (fd < 0 && errno == EEXIST)
        {
            // Name collision. Keep trying! 
            continue;
        }
        else if (fd < 0)
        {
            fprintf(stderr, "Couldn't create a temporary file: %s\n", strerror(errno));
            return "";
        }
        close(fd);
        return std::string(fname);
    }
    return "";
}

bool os::setEnvironmentVariable(std::string name, std::string value)
{
    std::string env = name + "=" + value;
    if (putenv(env.c_str()) == 0)
        return true;
    fprintf(stderr, "Failed to set environment variable %s: %s\n", name.c_str(), strerror(errno));
    return false;
}


bool os::clearEnvironmentVariable(std::string name)
{
    return setEnvironmentVariable(name, "");
}

bool os::getUserInformation(std::string &name, std::string &login, std::string &host)
{
    // TODO
    return true;
}

bool os::createDirectory(std::string name, int mode)
{
    int ret = _mkdir(name.c_str());
    if (ret < 0)
        return errno == EEXIST ? true : false;
    return ret == 0;
}

#ifdef ENABLE_SANDBOX
int os::syscallNameToNumber(std::string name)
{
    // TODO
    return 0;
}

bool os::initChildSandbox(std::string filterpath, const nlohmann::json &rules)
{
    // TODO
    return true;
}

bool os::initParentSandbox(std::string filterpath, const nlohmann::json &rules, pid_t tracee_pid)
{
    // TODO
    return true;
}
#endif // ENABLE_SANDBOX

#endif // __MINGW64__
