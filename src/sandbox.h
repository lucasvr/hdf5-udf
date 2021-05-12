/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: sandbox.h
 *
 * Sandboxing routines to prevent certain system calls
 * from being executed by the user-defined-functions.
 */
#ifndef __sandbox_h
#define __sandbox_h

#include <stdbool.h>
#include <pcrecpp.h>
#include <sys/user.h>
#include <seccomp.h>
#include <functional>
#include <algorithm>
#include <vector>
#include <deque>
#include <string>
#include "sharedlib_manager.h"
#include "json.hpp"

using json = nlohmann::json;

enum fs_access_mode {
    READONLY = 0,
    READWRITE = 1,
};

class Sandbox {
public:
    Sandbox() {}
    ~Sandbox() {}

    // Entry point called by the child process (i.e., the one that runs the UDF).
    // This method loads Seccomp rules and configures the process so it waits
    // to be controlled by the parent with Ptrace.
    bool initChild(std::string filterpath, const json &rules);

    // Entry point called by the parent process. This method scans the config
    // file to retrieve the filesystem paths allowed to be accessed by the child
    // process (i.e., the UDF) and monitors the system calls it executes with
    // Ptrace. Filesystem path violations are punished with SIGKILL.
    bool initParent(std::string filterpath, const json &rules, pid_t tracee_pid);

private:
    bool parseFilesystemRules(const json &rules);

    bool parseSyscallRules(const json &rules, scmp_filter_ctx &ctx);

    bool monitorSystemCalls(pid_t tracee_pid, bool &fatal);

    bool allowedFilesystemAccess(
        pid_t tracee_pid,
        const struct user_regs_struct &regs,
        std::string &path);

    bool havePermission(std::string path, bool rw_wanted);

    std::string syscallPathArgument(pid_t pid, long arg0);

    std::string syscallPathArgument(pid_t pid, int at_fd, long arg1);

    // List of regular expressions representing filesystem paths that
    // the UDF is allowed to access
    std::deque<std::tuple<pcrecpp::RE, fs_access_mode>> fs_access_rules;
};

#endif