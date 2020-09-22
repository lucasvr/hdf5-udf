/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: sandbox.cpp
 *
 * Sandboxing routines to prevent certain system calls
 * from being executed by the user-defined-functions.
 */
#include <stdio.h>
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
#include <algorithm>
#include <vector>
#include <string>
#include "sandbox.h"

Sandbox::Sandbox()
{
    struct stat statbuf;
    std::string path = "libhdf5_udf-wrappers.so";
    if (stat("libhdf5_udf-wrappers.so", &statbuf) == 0)
    {
        // We are running from the distribution source tree
        path = "./libhdf5_udf-wrappers.so";
    }
    wrapperh = dlopen(path.c_str(), RTLD_NOW);
}

Sandbox::~Sandbox()
{
    if (wrapperh)
        dlclose(wrapperh);
}

#define ALLOW(syscall, ...) do { \
    if (seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(syscall), __VA_ARGS__) < 0) { \
        fprintf(stderr, "Failed to configure seccomp rule for '" #syscall "' syscall\n"); \
        return false; \
    } \
} while (0)

bool Sandbox::loadRules()
{
    scmp_filter_ctx ctx = seccomp_init(SCMP_ACT_KILL_PROCESS);

    // One particular use case of HDF5-UDF is to retrieve data
    // from servers exposed on the Internet and to provide that
    // data to the application using the HDF5 dataset interface.
    // The syscalls below should be sufficient to allow a program
    // to connect to an external host and communicate with it.
    // Any other syscall is denied by seccomp and will cause the
    // UDF thread to be killed.

    // Fundamental system calls we want to allow
    ALLOW(brk, 0);

    // Sockets-related system calls
    ALLOW(socket, 0);
    ALLOW(setsockopt, 0);
    ALLOW(ioctl, 1, SCMP_A1(SCMP_CMP_EQ, FIONREAD));
    ALLOW(connect, 0);
    ALLOW(select, 0);
    ALLOW(poll, 0);
    ALLOW(read, 0);
    ALLOW(recv, 0);
    ALLOW(recvfrom, 0);
    ALLOW(write, 0);
    ALLOW(send, 0);
    ALLOW(sendto, 0);
    ALLOW(sendmsg, 0);
    ALLOW(close, 0);

    // System calls issued by gethostbyname(). Some of these could be potentially
    // misused by malicious user-defined functions; we rely on syscall_wrappers.cpp
    // to check their string-based arguments to decide to allow or reject them.
    ALLOW(stat, 0);
    ALLOW(lstat, 0);
    ALLOW(fstat, 0);
    ALLOW(fstat64, 0);
    ALLOW(open, 1, SCMP_A1(SCMP_CMP_MASKED_EQ, O_RDONLY, O_RDONLY));
    ALLOW(openat, 1, SCMP_A2(SCMP_CMP_MASKED_EQ, O_RDONLY, O_RDONLY));
    ALLOW(mmap, 0);
    ALLOW(mmap2, 0);
    ALLOW(munmap, 0);
    ALLOW(lseek, 0);
    ALLOW(_llseek, 0);
    ALLOW(futex, 0);
    ALLOW(uname, 0);
    ALLOW(mprotect, 0);    

    // Load seccomp rules
    int ret = seccomp_load(ctx);
    seccomp_release(ctx);
    if (ret < 0)
        fprintf(stderr, "Failed to load seccomp filter: %s\n", strerror(errno));

    return ret == 0;
}