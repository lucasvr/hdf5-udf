/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: file_search.cpp
 *
 * Very simple glob()-like interfaces to search for files given a directory
 * path and file name patterns.
 */
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <functional>
#include "file_search.h"

static bool find_files(
    std::string dir,
    std::vector<std::string> &out,
    std::function<bool(std::string)> match, 
    bool subdirs)
{
    DIR *dp = opendir(dir.c_str());
    if (! dp)
    {
        // The only error we care about is EPERM. The caller creates the given directory
        // and retries the operation otherwise.
        return errno == EPERM ? false : true;
    }

    struct dirent *entry;
    while ((entry = readdir(dp)))
    {
        if (entry->d_name[0] == '.')
        {
            // Ignore hidden directories, dot, and dot-dot.
            continue;
        }
        std::string path = dir + "/" + std::string(entry->d_name);
        if (match(entry->d_name))
        {
            // Found a match
            out.push_back(path);
            continue;
        }
        // Sadly, MINGW doesn't provide a d_type member on struct dirent, so we have to
        // stat() the file to tell its type. We could place an ifdef here and save a trip
        // to the kernel on Linux and MacOS, but it's best to keep the code clean.
        struct stat statbuf;
        if (subdirs == true && stat(path.c_str(), &statbuf) == 0 && S_ISDIR(statbuf.st_mode))
        {
            // Search one level down
            find_files(path, out, match, false);
        }
    }
    closedir(dp);
    return true;
}

bool findByExtension(std::string dir, std::string ext, std::vector<std::string> &out, bool subdirs)
{
    auto string_ends_with = [](std::string input, std::string suffix) {
        return input.size() >= suffix.size() &&
            input.compare(input.size()-suffix.size(), suffix.size(), suffix) == 0;
    };

    auto _matchFunction = [&](std::string fname) {
         return string_ends_with(fname, ext);
    };

    return find_files(dir, out, _matchFunction, subdirs);
}

bool findByPattern(std::string dir, std::string pattern, std::vector<std::string> &out, bool subdirs)
{
    std::string start = pattern.substr(0, pattern.find("*"));
    std::string end = pattern.substr(pattern.find("*")+1, pattern.npos);
    printf("findByPattern: start=%s, end=%s\n", start.c_str(), end.c_str());

    auto string_starts_with = [](std::string input, std::string prefix) {
        return input.size() >= prefix.size() &&
            input.compare(0, prefix.size(), prefix) == 0;
    };

    auto string_ends_with = [](std::string input, std::string suffix) {
        return input.size() >= suffix.size() &&
            input.compare(input.size()-suffix.size(), suffix.size(), suffix) == 0;
    };

	auto _matchFunction = [&](std::string fname) {
         return string_starts_with(fname, start) && string_ends_with(fname, end);
    };

    return find_files(dir, out, _matchFunction, subdirs);
}
