/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: sandbox_linux.cpp
 *
 * High-level interfaces for system call interception on Linux.
 */
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <limits.h>
#include <syscall.h>
#include <signal.h>
#include <sstream>
#include "sandbox.h"
#include "sysdefs.h"
#include "sandbox_linux.h"

#ifndef SYS_SECCOMP
#define SYS_SECCOMP 1
#endif

using json = nlohmann::json;

////////////////////////////
// Tracer (parent) process
////////////////////////////

bool LinuxSandbox::initParent(std::string filterpath, const json &rules, pid_t tracee_pid)
{
    if (rules.contains("filesystem"))
    {
        // Iterate over the rules to configure the filesystem path checker.
        // Note that the rules have been already validated by the time this
        // function is called.
        parseFilesystemRules(rules);
    }

    // Monitor system calls issued by the child process
    bool fatal = true;
    if (monitorSystemCalls(tracee_pid, fatal) == false)
    {
        if (fatal)
        {
            kill(tracee_pid, SIGKILL);
            waitpid(tracee_pid, 0, 0);
        }
        return false;
    }
    return true;
}

bool LinuxSandbox::parseFilesystemRules(const json &rules)
{
    for (auto &element: rules["filesystem"].items())
        for (auto &fs_element: element.value().items())
        {
            auto name = fs_element.key();
            auto rule = fs_element.value();
            if (name.size() && name[0] == '#')
                continue;

            // Regex 'star' processing, borrowed from AppArmor's parser_regex.c
            std::string path;
            const char *sptr = name.c_str();
            while (*sptr)
            {
                if (*sptr == '*')
                {
                    if (path.length() && path[path.length()-1] == '/')
                    {
                        // Modify what's emmited for * and ** when used as the
                        // only path component. Examples:
                        // /*   /*/   /**/   /**
                        //
                        // This prevents these expressions from matching directories
                        // or invalid paths. In these cases, * and ** must match at
                        // least 1 character to get a valid path element. For instance:
                        // /foo/*    -> should not match /foo/
                        // /foo/*bar -> should match /foo/bar
                        // /*/foo    -> should not match //foo
                        const char *s = sptr;
                        while (*s == '*')
                            s++;
                        if (*s == '/' || !*s)
                            path.append("[^/]");
                    }
                    if (*(sptr + 1) == '*')
                        path.append(".*");
                    else
                        path.append("[^/]*");
                }
                else
                    path.push_back(*sptr);
                sptr++;
            }

            auto access_mode = rule.get<std::string>();
            pcrecpp::RE re(path);
            fs_access_rules.push_front(std::make_tuple(
                std::move(re),
                access_mode.compare("ro") == 0 ? READONLY : READWRITE
            ));
        }

    return true;
}

bool LinuxSandbox::monitorSystemCalls(pid_t pid, bool &fatal)
{
    int ret, status;

    // Failure to monitor the child process result in a pkill
    fatal = true;

    // Wait until the child returns from PTRACE_TRACEME + SIGSTOP
    waitpid(pid, &status, __WALL);
    if (WIFEXITED(status))
    {
        fatal = false; // already dead
        return false;
    }

    // Bring the child process down with us if we die, plus chase any subprocesses its spawns.
    ret = ptrace(PTRACE_SETOPTIONS, pid, NULL,
        PTRACE_O_EXITKILL |
        PTRACE_O_TRACECLONE |
        PTRACE_O_TRACEEXEC |
        PTRACE_O_TRACEFORK |
        PTRACE_O_TRACEVFORK |
        PTRACE_O_TRACESECCOMP);
    if (ret < 0)
    {
        fprintf(stderr, "Failed to configure ptrace on parent: %s\n", strerror(errno));
        return false;
    }

    while (true)
    {
        // Wait for system call (entry point)
        ret = ptrace(PTRACE_SYSCALL, pid, NULL, NULL);
        if (ret < 0)
        {
            fprintf(stderr, "Failed to trace system call: %s\n", strerror(errno));
            return false;
        }
        waitpid(pid, &status, __WALL);
        if (WIFEXITED(status))
            break;

        // Get system call information
        struct user_regs_struct regs;
        ret = ptrace(PTRACE_GETREGS, pid, NULL, &regs);
        if (ret < 0) {
            fprintf(stderr, "Failed to get system call registers: %s\n", strerror(errno));
            return false;
        }

        std::string path;
        if (! allowedFilesystemAccess(pid, regs, path))
        {
            // Invalid filesystem access: kill the UDF
            fprintf(stderr, "UDF attempted to access blocked filesystem path '%s'\n", path.c_str());
            return false;
        }

        // Wait for system call (exit point)
        ret = ptrace(PTRACE_SYSCALL, pid, NULL, NULL);
        if (ret < 0)
        {
            fprintf(stderr, "Failed to trace system call: %s\n", strerror(errno));
            return false;
        }
        waitpid(pid, &status, __WALL);
        if (WIFEXITED(status))
            break;
    }

    fatal = false;
    return WEXITSTATUS(status) == 0;
}

#if __WORDSIZE == 64
# define SYSCALL_NR(reg) reg.orig_rax
#else
# define SYSCALL_NR(reg) reg.orig_eax
#endif

#define SYSCALL_ARG0(reg) regs.rdi
#define SYSCALL_ARG1(reg) regs.rsi
#define SYSCALL_ARG2(reg) regs.rdx
#define SYSCALL_ARG3(reg) regs.r10
#define SYSCALL_ARG4(reg) regs.r8
#define SYSCALL_ARG5(reg) regs.r9

bool LinuxSandbox::allowedFilesystemAccess(
    pid_t pid,
    const struct user_regs_struct &regs,
    std::string &path)
{
    long syscall_nr = SYSCALL_NR(regs);
    long arg0 = SYSCALL_ARG0(regs);
    long arg1 = SYSCALL_ARG1(regs);
    long arg2 = SYSCALL_ARG2(regs);
    int flags;

    switch (syscall_nr)
    {
        case SYS_open:
            path = syscallPathArgument(pid, arg0);
            flags = (int) arg1;
            return havePermission(path, (arg1 & O_ACCMODE) != O_RDONLY);

        case SYS_openat:
            path = syscallPathArgument(pid, arg0, arg1);
            flags = (int) arg2;
            return havePermission(path, (flags & O_ACCMODE) != O_RDONLY);

        case SYS_symlinkat:
            path = syscallPathArgument(pid, arg0);
            if (! havePermission(path, true))
                return false;
            path = syscallPathArgument(pid, arg1, arg2);
            return havePermission(path, true);

        case SYS_link:
        case SYS_symlink:
        case SYS_rename:
            path = syscallPathArgument(pid, arg0);
            if (! havePermission(path, true))
                return false;
            path = syscallPathArgument(pid, arg1);
            return havePermission(path, true);

        case SYS_mkdir:
        case SYS_rmdir:
        case SYS_creat:
        case SYS_unlink:
        case SYS_chmod:
        case SYS_chown:
        case SYS_lchown:
        case SYS_truncate:
        case SYS_mknod:
        case SYS_swapon:
        case SYS_swapoff:
        case SYS_setxattr:
        case SYS_lsetxattr:
        case SYS_removexattr:
        case SYS_lremovexattr:
            path = syscallPathArgument(pid, arg0);
            return havePermission(path, true);

        case SYS_readlink:
        case SYS_utime:
        case SYS_utimes:
        case SYS_acct:
        case SYS_stat:
        case SYS_lstat:
        case SYS_statfs:
        case SYS_getxattr:
        case SYS_lgetxattr:
        case SYS_listxattr:
        case SYS_llistxattr:
            path = syscallPathArgument(pid, arg0);
            return havePermission(path, false);

        default:
            return true;
    }
}

std::string LinuxSandbox::syscallPathArgument(pid_t pid, long arg0)
{
    // arg0 = memory address (pid space) of the path
    std::string path;
    long address = arg0;

    // PTRACE_PEEKDATA outputs a long (8 bytes) starting at the requested address.
    // We use a union so we can easily convert that long integer into a byte stream
    // to process pathnames.
    union u {
        long value;
        char bytes[sizeof(long)];
    } peekdata;

    // PTRACE_PEEKDATA must use aligned addresses, or it may give us -EIO.
    long aligned = address - (address % sizeof(long));
    if (aligned != 0)
    {
        errno = 0;
        peekdata.value = ptrace(PTRACE_PEEKDATA, pid, aligned, NULL);
        if (peekdata.value < 0 && errno != 0)
        {
            fprintf(stderr, "Failed to peek child data: %s\n", strerror(errno));
            return "";
        }

        long start_offset = address - aligned;
        path.append(&peekdata.bytes[start_offset], sizeof(long)-start_offset);

        // Stop here if we already have a NULL terminator
        for (uint64_t i=start_offset; i<sizeof(long); ++i)
            if (peekdata.bytes[i] == '\0')
                return path;

        // Adjust the next address and keep going
        address = aligned + sizeof(long);
    }

    while (true)
    {
        errno = 0;
        peekdata.value = ptrace(PTRACE_PEEKDATA, pid, address, NULL);
        if (peekdata.value < 0 && errno != 0)
        {
            fprintf(stderr, "Failed to peek child data: %s\n", strerror(errno));
            return "";
        }

        // Stop here if we already have a NULL terminator
        for (uint64_t i=0; i<sizeof(long); ++i)
            if (peekdata.bytes[i] == '\0')
            {
                path.append(peekdata.bytes, i);
                return path;
            }

        path.append(peekdata.bytes, sizeof(long));
        address += sizeof(long);
    }

    // Never reached
    return "";
}

std::string LinuxSandbox::syscallPathArgument(pid_t pid, int at_fd, long arg1)
{
    // at_fd = "at" file descriptor
    // arg1 = memory address (pid space) of the path (potentially relative to arg0)
    if (at_fd == AT_FDCWD)
        return syscallPathArgument(pid, arg1);

    // Get the path associated with arg0
    char buf[PATH_MAX];
    std::stringstream ss;
    ss << "/proc/" << pid << "/fd/" << at_fd;
    ssize_t n = readlink(ss.str().c_str(), buf, sizeof(buf)-1);
    if (n < 0)
    {
        fprintf(stderr, "Failed to read %s: %s\n", ss.str().c_str(), strerror(errno));
        return "";
    }
    buf[n] = '\0';

    // Concatenate arg0 with arg1
    std::string path1(buf);
    std::string path2 = syscallPathArgument(pid, arg1);
    return path1 + "/" + path2;
}

bool LinuxSandbox::havePermission(std::string path, bool rw_wanted)
{
    // Test the path against the regular expressions loaded from the
    // configuration file.
    for (auto &rule: fs_access_rules)
    {
        pcrecpp::RE &re = std::get<0>(rule);
        bool allowed = std::get<1>(rule);
        if (re.FullMatch(path) && (! rw_wanted || (rw_wanted && allowed == READWRITE)))
            return true;
    }
    return false;
};

///////////////////////////
// Tracee (child) process
///////////////////////////

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

bool LinuxSandbox::initChild(std::string filterpath, const json &rules)
{
    // Do not grant new privileges on execve()
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0)
    {
        fprintf(stderr, "Failed to set PR_SET_NO_NEW_PRIVS: %s\n", strerror(errno));
        return false;
    }

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

    // Iterate over the syscall rules to configure Seccomp.
    // Note that the rules have been already validated by the time this function is called.
    if (rules.contains("syscalls"))
        parseSyscallRules(rules, ctx);

    // Let this process be traced by its parent and wait until parent attaches to us
    if (ptrace(PTRACE_TRACEME, 0, 0, 0) < 0)
    {
        fprintf(stderr, "Failed to configure ptrace on child: %s\n", strerror(errno));
        return false;
    }
    raise(SIGSTOP);

    // Load seccomp rules
    int ret = seccomp_load(ctx);
    seccomp_release(ctx);
    if (ret < 0)
        fprintf(stderr, "Failed to load seccomp filter: %s\n", strerror(errno));

    return ret == 0;
}

#define ALLOW(syscall_nr, ...) do { \
    if (seccomp_rule_add(ctx, SCMP_ACT_ALLOW, syscall_nr, __VA_ARGS__) < 0) { \
        fprintf(stderr, "Failed to configure seccomp rule for syscall '%s'\n", name.c_str()); \
        return false; \
    } \
} while (0)

bool LinuxSandbox::parseSyscallRules(const json &rules, scmp_filter_ctx &ctx)
{
    for (auto &element: rules["syscalls"].items())
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

    return true;
}