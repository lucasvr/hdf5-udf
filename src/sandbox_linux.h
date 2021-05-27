/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: sandbox_linux.h
 *
 * Sandboxing routines to prevent certain system calls
 * from being executed by the user-defined-functions.
 */
#ifndef __sandbox_linux_
#define __sandbox_linux_h

#include <seccomp.h>
#include "sandbox.h"

class LinuxSandbox : public Sandbox {
public:
    LinuxSandbox() {}
    ~LinuxSandbox() {}

    // Entry point called by the child process (i.e., the one that runs the UDF).
    // This method loads Seccomp rules and configures the process so it waits
    // to be controlled by the parent with Ptrace.
    bool initChild(std::string filterpath, const nlohmann::json &rules);

    // Entry point called by the parent process. This method scans the config
    // file to retrieve the filesystem paths allowed to be accessed by the child
    // process (i.e., the UDF) and monitors the system calls it executes with
    // Ptrace. Filesystem path violations are punished with SIGKILL.
    bool initParent(std::string filterpath, const nlohmann::json &rules, pid_t tracee_pid);

private:
    bool parseFilesystemRules(const nlohmann::json &rules);

    bool parseSyscallRules(const nlohmann::json &rules, scmp_filter_ctx &ctx);

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