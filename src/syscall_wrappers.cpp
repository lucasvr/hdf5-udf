/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: syscall_wrappers.cpp
 *
 * Seccomp does not dereference pointers, so string-based rules are a no-go.
 * This file provides wrappers for system calls whose string-based arguments
 * we want to check (LD_PRELOAD-style).
 *
 * Note that this code is compiled into a shared library. We override syscalls
 * of interest by linking the UDF with the syscall_intercept library.
 */
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <syscall.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <glob.h>
#include <libsyscall_intercept_hook_point.h>
#include <algorithm>
#include <string>
#include <vector>

extern "C" {

static std::vector<std::string> files_allowed, dirs_allowed;

static void print_syscall(std::string name, long arg)
{
    char *path = (char *) arg;
    syscall_no_intercept(SYS_write, 2, name.c_str(), name.size());
    syscall_no_intercept(SYS_write, 2, " ", 1);
    syscall_no_intercept(SYS_write, 2, path, strlen(path));
    syscall_no_intercept(SYS_write, 2, "\n", 1);   
}

static int test_file_ok(long arg, long *ret)
{
    char *path = (char *) arg;
    for (auto &p: files_allowed)
        if (p.compare(path) == 0)
            return 1;
    *ret = -EPERM;
    return 0;
}

// A non-zero return value returned by the callback function is used to signal
// to the intercepting library that the specific system call was ignored by the
// user and the original syscall should be executed. A zero return value signals
// that the user takes over the system call. 
static int hook(long syscall_nr, long arg0, long arg1, long arg2, long arg3, long arg4, long arg5, long *ret)
{
	(void) arg2;
	(void) arg3;
	(void) arg4;
	(void) arg5;
    switch (syscall_nr)
    {
        case SYS_stat:
        case SYS_lstat:
        case SYS_open:
            return test_file_ok(arg0, ret);
        case SYS_openat:
            return test_file_ok(arg1, ret);
        case SYS_fstat:
        default:
            return 1;
    }
}

static __attribute__((constructor)) void init()
{
    const char *files[] = {
        "/etc/resolv.conf",
#if 0
        "/etc/ld.so.cache",
        "/etc/nsswitch.conf",
        "/lib/libc.so*",
        "/lib/ld-linux*.so*",
        "/lib/libresolv.so*",
        "/lib/libnss_files.so*",
#endif
        NULL,
    };

    // Resolve wildcards in the files[] array
    for (auto i=0; files[i]; ++i)
    {
        // Update list of files the program is allowed to open
        const char *path = files[i];
        if (strstr(path, "*") == NULL)
            files_allowed.push_back(path);
        else
        {
            glob_t globbuf;
            if (glob(path, GLOB_NOSORT, NULL, &globbuf) != 0)
                continue;
            for (size_t n=0; n < globbuf.gl_pathc; ++n)
                files_allowed.push_back(globbuf.gl_pathv[n]);
            globfree(&globbuf);
        }

        // Update list of directories the program is allowed to open
        auto last = files_allowed.back();
        auto sep = last.substr(1).find("/");
        if (sep != std::string::npos)
        {
            auto lastdir = last.substr(0, sep+1);
            auto res = std::find(
                    dirs_allowed.begin(),
                    dirs_allowed.end(),
                    lastdir);
            if (res == dirs_allowed.end())
                dirs_allowed.push_back(lastdir);
        }
    }

    // Set up our system call hook
    intercept_hook_point = &hook;
}

} /* extern "C" */