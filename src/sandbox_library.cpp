/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: sandbox_library.cpp
 *
 * Seccomp does not dereference pointers, so string-based rules are a no-go.
 * This file provides wrappers for system calls whose string-based arguments
 * we want to check (LD_PRELOAD-style).
 *
 * Note that this code is compiled into a shared library. The syscall_intercept
 * library modifies the resulting shared library's constructor so that Glibc
 * system calls can be intercepted and their arguments checked by ourselves.
 */
#include <stdio.h>
#include <stdbool.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <glob.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <seccomp.h>
#include <syscall.h>
#include <signal.h>
#include <libsyscall_intercept_hook_point.h>
#include <algorithm>
#include <vector>
#include <string>
#include "sysdefs.h"
#include "json.hpp"

#ifndef SYS_SECCOMP
#define SYS_SECCOMP 1
#endif

using json = nlohmann::json;

// List of files allowed to be accessed by the UDF
static std::vector<std::string> files_allowed;

extern "C" {

////////////////////////////////
// syscall-intercept interface
////////////////////////////////

static void __attribute__((unused)) print_syscall(std::string name, long arg)
{
    char *path = (char *) arg;
    syscall_no_intercept(SYS_write, 2, name.c_str(), name.size());
    syscall_no_intercept(SYS_write, 2, " ", 1);
    syscall_no_intercept(SYS_write, 2, path, strlen(path));
    syscall_no_intercept(SYS_write, 2, "\n", 1);   
}

static int syscall_intercept_hook(
    long syscall_nr,
    long arg0, long arg1, long arg2, long arg3, long arg4, long arg5,
    long *ret)
{
    (void) arg2;
    (void) arg3;
    (void) arg4;
    (void) arg5;
 
    auto test_file_ok = [&](long arg)
    {
        char *path = (char *) arg;
        for (auto &p: files_allowed)
            if (p.compare(path) == 0)
                return 1;
        *ret = -EACCES;
        return 0;
    };

    // A non-zero return value returned by the callback function is used to signal
    // to the intercepting library that the specific system call was ignored by the
    // user and the original syscall should be executed. A zero return value signals
    // that the user takes over the system call. 
    switch (syscall_nr)
    {
        case SYS_stat:
        case SYS_lstat:
        case SYS_open:
            return test_file_ok(arg0);
        case SYS_openat:
            return test_file_ok(arg1);
        case SYS_fstat:
        default:
            return 1;
    }
}

// Entry point, called from Sandbox::init()
bool sandbox_init_syscall_intercept(std::vector<std::string> paths_allowed)
{
    // Allow reading from CSV files in the current directory,
    // as long as they are not symlinks to other objects in
    // the filesystem. This rule is to be made configurable
    // in the next major release of HDF5-UDF.
    paths_allowed.push_back("*.csv");

    auto is_symlink = [&](const char *path)
    {
        struct stat statbuf;
        return lstat(path, &statbuf) == 0 && S_ISLNK(statbuf.st_mode);
    };

    // Resolve wildcards in paths_allowed[]
    for (auto &path: paths_allowed)
    {
        // Update list of files the program is allowed to open
        if (path.find("*") == std::string::npos)
        {
            if (! is_symlink(path.c_str()))
                files_allowed.push_back(path);
        }
        else
        {
            glob_t globbuf;
            if (glob(path.c_str(), GLOB_NOSORT, NULL, &globbuf) != 0)
                continue;
            for (size_t n=0; n < globbuf.gl_pathc; ++n)
                if (! is_symlink(globbuf.gl_pathv[n]))
                    files_allowed.push_back(globbuf.gl_pathv[n]);
            globfree(&globbuf);
        }
    }

    // Set up our system call hook
    intercept_hook_point = &syscall_intercept_hook;
    return true;
}

////////////////////////////////////
// System call filtering interface
////////////////////////////////////

#define ALLOW(syscall_nr, ...) do { \
    if (seccomp_rule_add(ctx, SCMP_ACT_ALLOW, syscall_nr, __VA_ARGS__) < 0) { \
        fprintf(stderr, "Failed to configure seccomp rule for syscall '%s'\n", name.c_str()); \
        return false; \
    } \
} while (0)

static void unallowed_syscall_handler(int signum, siginfo_t *info, void *ucontext)
{
    if (info->si_code == SYS_SECCOMP)
    {
        // Note: not every function is async-signal-safe (i.e., functions which can
        // be safely called within a signal handler). For instance, buffered I/O and
        // memory allocation are not -- and we use both here for the sake of convenience.
        // If the program gets interrupted by another signal while we call one of those
        // functions and the new signal handler also executes one of those functions
        // chances are that we'll end up with memory corruption. On the good side, we're
        // just about to kill the process anyway, so this may not hurt at all.
        char *name = seccomp_syscall_resolve_num_arch(info->si_arch, info->si_syscall);
        fprintf(stderr, "UDF attempted to execute blocked syscall %s\n", name);
        free(name);
        _exit(1);
    }
}

// Entry point, called from Sandbox::init()
bool sandbox_init_seccomp(const json &rules)
{
    // Let the process receive a SIGSYS when it executes a
    // system call that's not allowed.
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_flags = SA_SIGINFO;
    action.sa_sigaction = unallowed_syscall_handler;
    if (sigaction(SIGSYS, &action, NULL) < 0)
    {
        fprintf(stderr, "Failed setting a handler for SIGSYS: %s\n", strerror(errno));
        return false;
    }

    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGSYS);
    if (sigprocmask(SIG_UNBLOCK, &mask, NULL))
    {
        fprintf(stderr, "Failed to remove SIGSYS from list of blocked signals\n");
        return false;
    }

    scmp_filter_ctx ctx = seccomp_init(SCMP_ACT_TRAP);

    // Iterate over the rules to configure Seccomp.
    // Note that the rules have been already validated by the time this function is called.
    if (rules.contains("syscalls"))
    {
        for (auto &element: rules["syscalls"].items())
        {
            for (auto &syscall_element: element.value().items())
            {
                auto name = syscall_element.key();
                auto rule = syscall_element.value();
                if ((name.size() && name[0] == '#'))
                    continue;

                // Simple case: rule is a boolean
                auto syscall_nr = seccomp_syscall_resolve_name(name.c_str());
                if (rule.is_boolean())
                {
                    ALLOW(syscall_nr, 0);
                    continue;
                }

                // Rule has specific filters. At this point, any string-based rules
                // have been already converted into a number by the rule validation
                // code at SignatureHandler::validateProfileRules().
                auto rule_arg = rule["arg"].get<unsigned int>();
                auto rule_op = rule["op"].get<std::string>();

                unsigned long rule_value, rule_mask;
                if (rule["value"].is_string())
                {
                    auto value = rule["value"].get<std::string>();
                    rule_value = sysdefs.find(value)->second;
                }
                else
                    rule_value = rule["value"].get<unsigned long>();

                if (rule_op.compare("equals") == 0)
                {
                    // We currently support specifying a single argument only
                    ALLOW(syscall_nr, 1, SCMP_CMP64(rule_arg, SCMP_CMP_EQ, rule_value));
                    continue;
                }
                else if (rule_op.compare("masked_equals") == 0)
                {
                    if (rule["mask"].is_string())
                    {
                        auto value = rule["mask"].get<std::string>();
                        rule_mask = sysdefs.find(value)->second;
                    }
                    else
                        rule_mask = rule["mask"].get<unsigned long>();
                    ALLOW(syscall_nr, 1, SCMP_CMP64(rule_arg, SCMP_CMP_MASKED_EQ, rule_mask, rule_value));
                    continue;
                }
            }
        }
    }

    // Load seccomp rules
    int ret = seccomp_load(ctx);
    seccomp_release(ctx);
    if (ret < 0)
        fprintf(stderr, "Failed to load seccomp filter: %s\n", strerror(errno));

    return ret == 0;
}

} // extern "C"