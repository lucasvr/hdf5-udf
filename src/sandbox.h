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

class Sandbox {
public:
    Sandbox();
    ~Sandbox();
    bool loadRules();
private:
    void *wrapperh;
};

#endif