/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: sharedlib_manager.h
 *
 * Helper class to manage calls to dlopen and dlsym.
 */
#ifndef __sharedlib_manager_h
#define __sharedlib_manager_h 

#include <stdio.h>
#include <dlfcn.h>
#include <string>

class SharedLibraryManager {
public:
    SharedLibraryManager() :
        so_handle(NULL)
    {
    }

    ~SharedLibraryManager() {
        if (so_handle)
            dlclose(so_handle);
    }

    bool open(std::string so_file)
    {
        (void) dlerror();
        so_handle = dlopen(so_file.c_str(), RTLD_NOW);
        if (! so_handle)
            fprintf(stderr, "Failed to load %s: %s\n", so_file.c_str(), dlerror());
        return so_handle != NULL;
    }

    void *loadsym(std::string name)
    {
        (void) dlerror();
        void *symbol = dlsym(so_handle, name.c_str());
        if (! symbol)
            fprintf(stderr, "%s\n", dlerror());
        return symbol;
    }

private:
    void *so_handle;
};

#endif /* __sharedlib_manager_h */