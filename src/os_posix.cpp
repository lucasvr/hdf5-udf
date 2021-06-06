/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: os_posix.cpp
 *
 * POSIX routines implemented on both Linux and macOS
 */
#if defined(__linux__) or defined(__APPLE__)
#include <sys/utsname.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <seccomp.h>
#include <unistd.h>
#include <dirent.h>
#include <string.h>
#include <pwd.h>
#include "os.h"

std::string os::makeTemporaryFile(std::string template_name, std::string extension)
{
    char path[PATH_MAX];
    char *tmp = getenv("TMPDIR") ? : (char *) "/tmp";
    snprintf(path, sizeof(path)-1, "%s/%s%s", tmp, template_name.c_str(), extension.c_str());
    int fd = mkstemps(path, extension.size());
    if (fd < 0)
    {
        fprintf(stderr, "Error creating temporary file.\n");
        return std::string("");
    }
    close(fd);
    return std::string(path);
}

bool os::setEnvironmentVariable(std::string name, std::string value)
{
    if (setenv(name.c_str(), value.c_str(), 1) == 0)
        return true;
    fprintf(stderr, "Failed to set environment variable %s: %s\n", name.c_str(), strerror(errno));
    return false;
}

bool os::clearEnvironmentVariable(std::string name)
{
    if (unsetenv(name.c_str()) == 0)
        return true;
    fprintf(stderr, "Failed to clear environment variable %s: %s\n", name.c_str(), strerror(errno));
    return false;
}

bool os::getUserInformation(std::string &name, std::string &login, std::string &host)
{
    struct utsname uts;
    memset(&uts, 0, sizeof(uts));
    uname(&uts);

    auto pw = getpwuid(getuid());
    name = pw ? (strlen(pw->pw_gecos) ? pw->pw_gecos : pw->pw_name) : "Unknown";
    login = pw ? std::string(pw->pw_name) : "user";
    host = std::string(uts.nodename);
    return true;
}

bool os::createDirectory(std::string name, int mode)
{
    return mkdir(name.c_str(), mode);
}

#endif // __linux__ or __APPLE__
