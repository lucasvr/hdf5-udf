/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: sandbox.cpp
 *
 * High-level interfaces to seccomp and system call interception for
 * path-based filtering.
 */
#include <stdio.h>
#include <stdbool.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <seccomp.h>
#include <syscall.h>
#include <signal.h>
#include <algorithm>
#include <vector>
#include <string>
#include "sandbox.h"
#include "sysdefs.h"
#include "json.hpp"

#ifndef SYS_SECCOMP
#define SYS_SECCOMP 1
#endif

using json = nlohmann::json;

bool Sandbox::init(
    std::string filterpath,
    const std::vector<std::string> &paths_allowed,
    const json &rules)
{
    bool ret = initSeccomp(rules);
    if (ret == false)
        fprintf(stderr, "Failed to configure sandbox\n");
    return ret;
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

bool Sandbox::initSeccomp(const json &rules)
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