/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: os_posix.cpp
 *
 * POSIX routines implemented on both Linux and macOS
 */
#if defined(__linux__) or defined(__APPLE__)
#include <sys/utsname.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <seccomp.h>
#include <unistd.h>
#include <dirent.h>
#include <string.h>
#include <fcntl.h>
#include <pwd.h>
#include "os.h"

std::string os::makeTemporaryFile(std::string template_name, std::string extension)
{
    char path[PATH_MAX];
    char *tmp = getenv("TMPDIR") ? : (char *) "/tmp";
    snprintf(path, sizeof(path)-1, "%s/%s%s", tmp, template_name.c_str(), extension.c_str());
    int fd = mkstemps(path, extension.size());
    if (fd < 0)
    {
        fprintf(stderr, "Error creating temporary file.\n");
        return std::string("");
    }
    close(fd);
    return std::string(path);
}

bool os::setEnvironmentVariable(std::string name, std::string value)
{
    if (setenv(name.c_str(), value.c_str(), 1) == 0)
        return true;
    fprintf(stderr, "Failed to set environment variable %s: %s\n", name.c_str(), strerror(errno));
    return false;
}

bool os::clearEnvironmentVariable(std::string name)
{
    if (unsetenv(name.c_str()) == 0)
        return true;
    fprintf(stderr, "Failed to clear environment variable %s: %s\n", name.c_str(), strerror(errno));
    return false;
}

bool os::getUserInformation(std::string &name, std::string &login, std::string &host)
{
    struct utsname uts;
    memset(&uts, 0, sizeof(uts));
    uname(&uts);

    auto pw = getpwuid(getuid());
    name = pw ? (strlen(pw->pw_gecos) ? pw->pw_gecos : pw->pw_name) : "Unknown";
    login = pw ? std::string(pw->pw_name) : "user";
    host = std::string(uts.nodename);
    return true;
}

bool os::createDirectory(std::string name, int mode)
{
    int ret = mkdir(name.c_str(), mode);
    if (ret < 0)
        return errno == EEXIST ? true : false;
    return ret == 0;
}

bool os::execCommand(char *program, char *args[], std::string *out)
{
    if (out)
    {
        int pipefd[2];
        if (pipe(pipefd) < 0)
        {
            fprintf(stderr, "Failed to create pipe\n");
            return false;
        }

        pid_t pid = fork();
        if (pid == 0)
        {
            // Child: runs command, outputs to pipe
            dup2(pipefd[1], STDOUT_FILENO);
            close(pipefd[0]);
            close(pipefd[1]);
            execvp(program, args);
        }
        else if (pid < 0)
        {
            fprintf(stderr, "Failed to execute '%s': %s\n", program, strerror(errno));
            return false;
        }
        else if (pid > 0)
        {
            // Parent: reads from pipe, concatenates data to 'out' string
            fcntl(pipefd[0], F_SETFL, O_NONBLOCK);
            while (true)
            {
                char buf[8192];
                ssize_t n = read(pipefd[0], buf, sizeof(buf));
                if (n < 0 && errno == EWOULDBLOCK)
                {
                    int exit_status = 0;
                    if (waitpid(pid, &exit_status, WNOHANG) == pid)
                    {
                        if (exit_status != 0)
                        {
                            fprintf(stderr, "Failed to run the C++ preprocessor\n");
                            close(pipefd[0]);
                            close(pipefd[1]);
                            return false;
                        }
                        break;
                    }
                    continue;
                }
                else if (n <= 0)
                    break;
                out->append(buf, n);
            }
            close(pipefd[0]);
            close(pipefd[1]);
        }
    }
    else
    {
        pid_t pid = fork();
        if (pid == 0)
            execvp(program, args);
        else if (pid > 0)
        {
            int exit_status;
            wait4(pid, &exit_status, 0, NULL);
        }
        else if (pid < 0)
        {
            fprintf(stderr, "Failed to execute '%s': %s\n", program, strerror(errno));
            return false;
        }
    }
    return true;
}

bool os::isWindows()
{
    return false;
}

#endif // __linux__ or __APPLE__
