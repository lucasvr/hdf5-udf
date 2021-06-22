/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: os_windows.cpp
 *
 * Windows-specific routines
 */
#ifdef __MINGW64__
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <string.h>
#include <direct.h>
#include <fcntl.h>
#include <errno.h>
#include <processthreadsapi.h> // GetCurrentProcess()
#include <libloaderapi.h>      // GetModuleHandleA()
#include <handleapi.h>         // CloseHandle()
#include <fileapi.h>           // GetFinalPathNameByHandleA()
#include <winternl.h>          // NtQueryObject()
#include <windows.h>           // VOLUME_NAME_NONE
#include <share.h>
#define SECURITY_WIN32
#include <secext.h>            // GetUserNameExA()
#include <io.h>
#include <regex>
#include <set>
#include "os.h"

// Windows protocols: NTSTATUS values
#define STATUS_INFO_LENGTH_MISMATCH 0xc0000004L

typedef DWORD (WINAPI *NtQueryObject_t)(HANDLE, DWORD, VOID *, DWORD, VOID *);
typedef DWORD (WINAPI *NtQuerySystemInformation_t)(ULONG, VOID *, ULONG, ULONG *);


static bool isFileHandle(const HANDLE obj)
{
    // NtQueryObject() needs to be retrieved dlsym()-style
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    NtQueryObject_t NtQueryObject;
    NtQueryObject = (NtQueryObject_t) GetProcAddress(ntdll, "NtQueryObject");

    ULONG info_len = 8192;
    PUBLIC_OBJECT_TYPE_INFORMATION *info = (PUBLIC_OBJECT_TYPE_INFORMATION *) malloc(info_len);
    if (! NT_SUCCESS(NtQueryObject(obj, ObjectTypeInformation, info, info_len, NULL)))
    {
        fprintf(stderr, "Failed to query object type information\n");
        free(info);
        return false;
    }

    bool ret = wcscmp(info->TypeName.Buffer, L"File") == 0;
    free(info);
    return ret;
}

std::vector<std::string> os::openedH5Files()
{
    std::vector<std::string> out;
    std::set<std::string> seen;
    DWORD status;

    // NtQuerySystemInformation() needs to be retrieved dlsym()-style
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    NtQuerySystemInformation_t NtQuerySysInfo;
    NtQuerySysInfo = (NtQuerySystemInformation_t) GetProcAddress(ntdll, "NtQuerySystemInformation");

    // Call NtQuerySystemInformation() until we have figured the minimum required buffer size
    SYSTEM_INFORMATION_CLASS handleinfo_class = SystemHandleInformation;
    ULONG handleinfo_len = 1024*1024;
    SYSTEM_HANDLE_INFORMATION *handleinfo = (SYSTEM_HANDLE_INFORMATION *) malloc(handleinfo_len);
    while ((status = NtQuerySysInfo(handleinfo_class, handleinfo, handleinfo_len, NULL)) == STATUS_INFO_LENGTH_MISMATCH)
    {
        handleinfo_len *= 2;
        handleinfo = (SYSTEM_HANDLE_INFORMATION *) realloc(handleinfo, handleinfo_len);
    }
    if (! NT_SUCCESS(status))
    {
        fprintf(stderr, "Failed to query system handles\n");
        free(handleinfo);
        return out;
    }

    // We now have system information at our disposal, so we can start looping over
    // the handles currently open on the system. We are only considered in handles
    // of the current process.
    HANDLE self = GetCurrentProcess();
    DWORD pid = GetCurrentProcessId();
    for (ULONG i=0; i<handleinfo->Count; ++i)
    {
        SYSTEM_HANDLE_ENTRY handle = handleinfo->Handle[i];
        HANDLE hval = (HANDLE) (intptr_t) handle.HandleValue;
        if (handle.OwnerPid == pid && isFileHandle(hval))
        {
            TCHAR path[MAX_PATH];
            memset(path, 0, sizeof(path));
            DWORD flags = VOLUME_NAME_NONE | FILE_NAME_OPENED;
            if (GetFinalPathNameByHandleA(hval, path, MAX_PATH, flags) >= MAX_PATH)
            {
                // Buffer is not large enough to hold the resolved file name
                continue;
            }
            if (seen.find(path) != seen.end() || strlen(path) == 0)
            {
                // We have seen this file before or the file name is not valid
                continue;
            }

            DWORD attrib = GetFileAttributesA(path);
            if (attrib == 0xffffffff)
            {
                // Not a real file name
                continue;
            }
            else if (attrib & FILE_ATTRIBUTE_NORMAL || attrib & FILE_ATTRIBUTE_ARCHIVE)
            {
                // At last!
                out.push_back(std::string(path));
            }
            seen.insert(path);
        }
    }

    CloseHandle(self);
    return out;
}

std::string os::sharedLibraryName(std::string name)
{
    return "lib" + name + ".dll";
}

std::string os::configDirectory()
{
    const char *home = getenv("USERPROFILE");
    if (home)
        return std::string(home) + "\\AppData\\Local\\HDF5-UDF\\";
    fprintf(stderr, "Error: %%USERPROFILE%% is not set, falling back to %%PWD%%\n");
    return ".hdf5-udf/";
}

std::string os::makeTemporaryFile(std::string template_name, std::string extension)
{
    char fname[template_name.size()+extension.size()+1];
    sprintf(fname, "%s%s", template_name.c_str(), extension.c_str());
    char *xxxxxx = strstr(fname, "XXXXXX");
    if (! xxxxxx)
    {
        fprintf(stderr, "Error: malformed string provided to makeTemporaryFile()\n");
        return "";
    }
    const char *letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    for (int i=0; i<0x0fffffff; ++i)
    {
        for (int j=0; j<6; ++j)
            xxxxxx[j] = letters[rand() % strlen(letters)];
        int fd = _sopen(fname, _O_RDWR | _O_CREAT | _O_EXCL, _SH_DENYNO, _S_IREAD | _S_IWRITE);
        if (fd < 0 && errno == EEXIST)
        {
            // Name collision. Keep trying! 
            continue;
        }
        else if (fd < 0)
        {
            fprintf(stderr, "Couldn't create a temporary file: %s\n", strerror(errno));
            return "";
        }
        close(fd);

        // Always produce absolute paths on Windows
        char abspath[MAX_PATH];
        memset(abspath, 0, sizeof(abspath));
        _fullpath(abspath, fname, sizeof(abspath)-1);
        return std::string(abspath);
    }
    return "";
}

bool os::setEnvironmentVariable(std::string name, std::string value)
{
    std::string env = name + "=" + value;
    if (putenv(env.c_str()) == 0)
        return true;
    fprintf(stderr, "Failed to set environment variable %s: %s\n", name.c_str(), strerror(errno));
    return false;
}


bool os::clearEnvironmentVariable(std::string name)
{
    return setEnvironmentVariable(name, "");
}

bool os::getUserInformation(std::string &name, std::string &login, std::string &host)
{
    // Fallback values for login and hostname, used to compose the user's email address
    const char *env_login = getenv("USERNAME");
    if (env_login)
        login = env_login;

    const char *env_hostname = getenv("HOSTNAME");
    if (env_hostname)
        host = env_hostname;

    // Retrieve user's email address
    DWORD email_len = 127;
    TCHAR email_buf[email_len+1];
    if (GetUserNameExA(NameUserPrincipal, email_buf, &email_len))
    {
        std::string email(email_buf);
        size_t pos = email.find_first_of('@');
        if (pos != std::string::npos)
        {
            login = email.substr(0, pos);
            host = email.substr(pos+1);
        }
    }

    // Retrieve user's full name
    DWORD name_len = 127;
    TCHAR name_buf[name_len+1];
    if (GetUserNameExA(NameDisplay, name_buf, &name_len))
        name = name_buf;

    return true;
}

bool os::createDirectory(std::string name, int mode)
{
    int ret = _mkdir(name.c_str());
    if (ret < 0)
        return errno == EEXIST ? true : false;
    return ret == 0;
}

bool os::execCommand(char *program, char *args[], std::string *out)
{
    if (out)
    {
        // Flatten args[]
        std::string flat_args = "";
        for (int i=0; args[i]; ++i)
            flat_args += std::string(args[i]) + (args[i+1] ? " " : "");

        // Create a pipe that executes the program
        char buf[256];
        FILE *fp = _popen(flat_args.c_str(), "rt");
        if (! fp)
        {
            fprintf(stderr, "Failed to execute '%s': %s\n", program, strerror(errno));
            return false;
        }

        // Save stdout to '*out' string
        while (fgets(buf, sizeof(buf), fp))
            out->append(std::string(buf));
        _pclose(fp);
    }
    else
    {
        if (_spawnvp(_P_WAIT, program, args) < 0)
        {
            fprintf(stderr, "Failed to execute %s: %s\n", program, strerror(errno));
            return false;
        }
    }
    return true;
}

bool os::isWindows()
{
   return true;
}

#ifdef ENABLE_SANDBOX
int os::syscallNameToNumber(std::string name)
{
    // TODO
    return 0;
}

bool os::initChildSandbox(std::string libpath, const nlohmann::json &rules)
{
    // TODO
    return true;
}

bool os::initParentSandbox(std::string libpath, const nlohmann::json &rules, pid_t tracee_pid)
{
    // TODO
    return true;
}
#endif // ENABLE_SANDBOX

#endif // __MINGW64__
