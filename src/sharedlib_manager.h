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

#ifdef __MINGW64__
# include <windows.h>
# include <winbase.h>
# include <libloaderapi.h>

# define shlib_load(name) LoadLibraryEx(name, NULL, LOAD_LIBRARY_SEARCH_APPLICATION_DIR)
# define shlib_loadsym(handle, name) GetProcAddress((HMODULE) handle, name)
# define shlib_unload(lib) FreeLibrary((HMODULE) lib)

static std::string shlib_error()
{
    DWORD err = GetLastError();
    LPVOID msgbuf;
    DWORD flags = FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_ALLOCATE_BUFFER;
    DWORD lang = MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT);
    DWORD msglen = FormatMessage(flags, NULL, err, lang, (LPTSTR) &msgbuf, 0, NULL);
    if (msglen)
    {
        LPCSTR msgstr = (LPCSTR) msgbuf;
        std::string errmsg(msgstr, msgstr+msglen);
        LocalFree(msgbuf);
        return errmsg;
    }
    return "(unknown error)";
}

#else
# define shlib_load(name) dlopen(name, RTLD_NOW)
# define shlib_loadsym(handle, name) dlsym(handle, name)
# define shlib_unload(lib) dlclose(lib)
# define shlib_error() std::string(dlerror())
#endif

class SharedLibraryManager {
public:
    SharedLibraryManager() :
        so_handle(NULL)
    {
    }

    ~SharedLibraryManager() {
        if (so_handle)
            shlib_unload(so_handle);
    }

    bool open(std::string so_file)
    {
        (void) dlerror();
        so_handle = (void *) shlib_load(so_file.c_str());
        if (! so_handle)
            fprintf(stderr, "Failed to load %s: %s\n", so_file.c_str(), shlib_error().c_str());
        return so_handle != NULL;
    }

    void *loadsym(std::string name)
    {
        (void) dlerror();
        void *symbol = (void *) shlib_loadsym(so_handle, name.c_str());
        if (! symbol)
            fprintf(stderr, "%s\n", shlib_error().c_str());
        return symbol;
    }

private:
    void *so_handle;
};

#endif /* __sharedlib_manager_h */
