/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: sandbox.cpp
 *
 * High-level interfaces to seccomp and syscall-intercept.
 */
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <elf.h>
#include "sandbox.h"
#include "backend.h"

#define SANDBOX_SECTION_NAME ".hdf5-udf-sandbox"

// Declare a dummy backend so we can call Backend::saveToDisk()
class DummyBackend : public Backend {
public:
    std::string name() { return ""; }
    std::string extension() { return ""; }
    std::string compile(std::string udf_file, std::string template_file) { return ""; }
    bool run(
        const std::string filterpath,
        const std::vector<DatasetInfo> input_datasets,
        const DatasetInfo output_dataset,
        const char *output_cast_datatype,
        const char *udf_blob,
        size_t udf_blob_size) { return true; }
    std::vector<std::string> udfDatasetNames(std::string udf_file) { return std::vector<std::string>(); }
};

bool Sandbox::init(std::string filterpath)
{
    // The sandbox library is stored in a special ELF section of the filter file.
    // We retrieve it from that section, save it to a temporary file and then
    // dlopen() that file so we can retrieve its symbols.
    auto so_file = extractSymbol(filterpath, SANDBOX_SECTION_NAME);
    if (so_file.size() == 0)
    {
        fprintf(stderr, "Failed to extract sandbox code from shared library\n");
        return false;
    }
    if (shlib.open(so_file) == false)
    {
        unlink(so_file.c_str());
        return false;
    }

    // Note that we delete the temporary file prior to the initialization of
    // the syscall filter, as the filter is unlikely to allow calls to unlink().
    bool ret = false;
    bool (*syscall_filter_init)() = (bool(*)()) shlib.loadsym("syscall_filter_init");
    unlink(so_file.c_str());

    if (syscall_filter_init)
    {
        ret = syscall_filter_init();
        if (ret == false)
            fprintf(stderr, "Failed to configure sandbox\n");
    }
    return ret;
}

std::string Sandbox::extractSymbol(std::string elf, std::string symbol_name)
{
    int fd = open(elf.c_str(), O_RDONLY);
    if (fd < 0)
    {
        fprintf(stderr, "Failed to open %s: %s\n", elf.c_str(), strerror(errno));
        return "";
    }
    struct stat statbuf;
    if (fstat(fd, &statbuf) < 0)
    {
        fprintf(stderr, "Failed to stat %s: %s\n", elf.c_str(), strerror(errno));
        close(fd);
        return "";
    }

    // ELF header
    Elf64_Ehdr header;
    read(fd, &header, sizeof(header));

    // ELF symbol table
    auto symbol_table_size = header.e_shnum * header.e_shentsize;
    Elf64_Shdr *symbol_table = (Elf64_Shdr *) malloc(symbol_table_size);
    pread(fd, symbol_table, symbol_table_size, header.e_shoff);

    // ELF string table index
    auto string_table_offset = symbol_table[header.e_shstrndx].sh_offset;
    auto string_table_size = symbol_table[header.e_shstrndx].sh_size;
    char *string_table = (char *) malloc(string_table_size);
    pread(fd, string_table, string_table_size, string_table_offset);

    std::string payload;
    for (auto i=0; i<header.e_shnum; ++i)
    {
        auto my_name = string_table + symbol_table[i].sh_name;
        auto my_size = symbol_table[i].sh_size;
        if (symbol_name.compare(my_name) == 0)
        {
            // Read the library that we have previously attached to this symbol
            payload.resize(my_size);
            pread(fd, (void *) payload.data(), my_size, symbol_table[i].sh_offset);
            break;
        }
    }

    free(string_table);
    free(symbol_table);
    close(fd);

    DummyBackend backend;
    auto so_file = backend.writeToDisk(payload.data(), payload.size(), ".so");
    if (so_file.size() == 0)
    {
        fprintf(stderr, "Failed to write payload to disk\n");
        return "";
    }
    chmod(so_file.c_str(), 0755);
    return so_file;
}