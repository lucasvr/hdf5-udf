/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: backend_lua.cpp
 *
 * Lua code parser and bytecode generation/execution.
 */
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include <errno.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <lua.hpp>
#include "config.h"
#include "udf_template_lua.h"
#include "backend_lua.h"
#include "anon_mmap.h"
#include "dataset.h"
#include "os.h"

/* Lua context */
static lua_State *State;

// Buffer used to hold the compound-to-struct name produced by luaGetCast()
static char compound_cast_name[256];

#define DATA_OFFSET(i)        (void *) (((char *) &State) + i)
#define NAME_OFFSET(i)        (void *) (((char *) &State) + 100 + i)
#define DIMS_OFFSET(i)        (void *) (((char *) &State) + 200 + i)
#define TYPE_OFFSET(i)        (void *) (((char *) &State) + 300 + i)
#define CAST_OFFSET(i)        (void *) (((char *) &State) + 400 + i)
#define SIZE_OFFSET(i)        (void *) (((char *) &State) + 500 + i)

extern "C" int index_of(const char *element)
{
    for (int index=0; index<100; ++index) {
        /* Set register key to get datasets name vector */
        lua_pushlightuserdata(State, NAME_OFFSET(index));
        lua_gettable(State, LUA_REGISTRYINDEX);
        const char *name = lua_tostring(State, -1);
        if (! strcmp(name, element))
            return index;
        else if (strlen(name) == 0)
            break;
    }
    fprintf(stderr, "Error: dataset %s not found\n", element);
    return -1;
}

/* Functions exported to the Lua template library */
extern "C" void *luaGetData(const char *element)
{
    int index = index_of(element);
    if (index >= 0)
    {
        /* Get datasets contents */
        lua_pushlightuserdata(State, DATA_OFFSET(index)); 
        lua_gettable(State, LUA_REGISTRYINDEX);
        return lua_touserdata(State, -1);
    }
    return NULL;
}

extern "C" int luaGetElementSize(const char *element)
{
    int index = index_of(element);
    if (index >= 0)
    {
        /* Get datatype size */
        lua_pushlightuserdata(State, SIZE_OFFSET(index));
        lua_gettable(State, LUA_REGISTRYINDEX);
        return (int) lua_tonumber(State, -1);
    }
    return 0;
}

extern "C" const char *luaGetType(const char *element)
{
    int index = index_of(element);
    if (index >= 0)
    {
        /* Set register key to get dataset type */
        lua_pushlightuserdata(State, TYPE_OFFSET(index));
        lua_gettable(State, LUA_REGISTRYINDEX);
        return lua_tostring(State, -1);
    }
    return NULL;
}

extern "C" const char *luaGetCast(const char *element)
{
    int index = index_of(element);
    if (index >= 0)
    {
        /* Set register key to get dataset type */
        lua_pushlightuserdata(State, CAST_OFFSET(index));
        lua_gettable(State, LUA_REGISTRYINDEX);
        const char *cast = lua_tostring(State, -1);
        if (! strcmp(cast, "void*") || ! strcmp(cast, "char*"))
        {
            // Cast compound structure
            LuaBackend backend;
            memset(compound_cast_name, 0, sizeof(compound_cast_name));
            snprintf(compound_cast_name, sizeof(compound_cast_name)-1,
                "struct %s_t *", backend.sanitizedName(element).c_str());
            return compound_cast_name;
        }
        return cast;
    }
    return NULL;
}

extern "C" const char *luaGetDims(const char *element)
{
    int index = index_of(element);
    if (index >= 0)
    {
        /* Set register key to get dataset size */
        lua_pushlightuserdata(State, DIMS_OFFSET(index));
        lua_gettable(State, LUA_REGISTRYINDEX);
        return lua_tostring(State, -1);
    }
    return NULL;
}

/* This backend's name */
std::string LuaBackend::name()
{
    return "LuaJIT";
}

/* Extension managed by this backend */
std::string LuaBackend::extension()
{
    return ".lua";
}

/* Compile Lua to bytecode using LuaJIT. Returns the bytecode as a string. */
std::string LuaBackend::compile(
    std::string udf_file,
    std::string compound_declarations,
    std::string &source_code,
    std::vector<DatasetInfo> &datasets)
{
    AssembleData data = {
        .udf_file                 = udf_file,
        .template_string          = std::string((char *) udf_template_lua),
        .compound_placeholder     = "// compound_declarations_placeholder",
        .compound_decl            = compound_declarations,
        .methods_decl_placeholder = "",
        .methods_decl             = "",
        .methods_impl_placeholder = "",
        .methods_impl             = "",
        .callback_placeholder     = "-- user_callback_placeholder",
        .extension                = this->extension()
    };
    auto lua_file = Backend::assembleUDF(data);
    if (lua_file.size() == 0)
    {
        fprintf(stderr, "Will not be able to compile the UDF code\n");
        return "";
    }

    std::string output = udf_file + ".bytecode";
    char *cmd[] = {
        (char *) "luajit",
        (char *) "-O3",
        (char *) "-b",
        (char *) lua_file.c_str(),
        (char *) output.c_str(),
        (char *) NULL
    };
    if (os::execCommand(cmd[0], cmd, NULL) == false)
    {
        fprintf(stderr, "Failed to build UDF\n");
        unlink(lua_file.c_str());
        return "";
    }

    struct stat statbuf;
    std::string bytecode;
    if (stat(output.c_str(), &statbuf) == 0) {
        std::ifstream data(output, std::ifstream::binary);
        std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(data), {});
        bytecode.assign(buffer.begin(), buffer.end());
        data.close();

        unlink(output.c_str());
    }

    // Read source file
    std::ifstream ifs(lua_file.c_str());
    source_code = std::string((std::istreambuf_iterator<char>(ifs)),
        (std::istreambuf_iterator<char>()));
    ifs.close();

    unlink(lua_file.c_str());
    return bytecode;
}

/* Execute the user-defined-function embedded in the given bytecode */
bool LuaBackend::run(
    const std::string libpath,
    const std::vector<DatasetInfo> &input_datasets,
    const DatasetInfo &output_dataset,
    const char *output_cast_datatype,
    const char *bytecode,
    size_t bytecode_size,
    const json &rules)
{
    lua_State *L = luaL_newstate();
    State = L;

    lua_pushcfunction(L, luaopen_base);
    lua_call(L,0,0);
    lua_pushcfunction(L, luaopen_math);
    lua_call(L,0,0);
    lua_pushcfunction(L, luaopen_string);
    lua_call(L,0,0);
    lua_pushcfunction(L, luaopen_ffi);
    lua_call(L,0,0);
    lua_pushcfunction(L, luaopen_jit);
    lua_call(L,0,0);
    lua_pushcfunction(L, luaopen_package);
    lua_call(L,0,0);
    lua_pushcfunction(L, luaopen_table);
    lua_call(L,0,0);

    // We want to make the output dataset writeable by the UDF. Because
    // the UDF is run under a separate process we have to use a shared
    // memory segment which both processes can read and write to.
    size_t room_size = output_dataset.getGridSize() * output_dataset.getStorageSize();
    AnonymousMemoryMap mm(room_size);
    if (! mm.createMapFor(output_dataset.data))
        return false;

    // Let output_dataset.data point to the shared memory segment
    DatasetInfo output_dataset_copy(output_dataset);
    output_dataset_copy.data = mm.mm;

    // index_of() uses 'DatasetInfo.name.size() == 0' as stop condition,
    // so we create a dummy entry, with an empty name, that gets pushed
    // into the end of the dataset_info vector.
    DatasetInfo empty_entry("", std::vector<hsize_t>(), "", -1);
    std::vector<DatasetInfo> dataset_info;
    dataset_info.push_back(std::move(output_dataset_copy));
    dataset_info.insert(
        dataset_info.end(), input_datasets.begin(), input_datasets.end());
    dataset_info.push_back(std::move(empty_entry));

    /* Populate vector of dataset names, sizes, and types */
    for (size_t i=0; i<dataset_info.size(); ++i)
    {
        /* Data */
        lua_pushlightuserdata(L, DATA_OFFSET(i));
        lua_pushlightuserdata(L, (void *) dataset_info[i].data);
        lua_settable(L, LUA_REGISTRYINDEX);

        /* Name */
        lua_pushlightuserdata(L, NAME_OFFSET(i));
        lua_pushstring(L, dataset_info[i].name.c_str());
        lua_settable(L, LUA_REGISTRYINDEX);

        /* Dimensions */
        lua_pushlightuserdata(L, DIMS_OFFSET(i));
        lua_pushstring(L, dataset_info[i].dimensions_str.c_str());
        lua_settable(L, LUA_REGISTRYINDEX);

        /* Type */
        lua_pushlightuserdata(L, TYPE_OFFSET(i));
        lua_pushstring(L, dataset_info[i].getDatatypeName());
        lua_settable(L, LUA_REGISTRYINDEX);

        /* Type, used for casting purposes */
        lua_pushlightuserdata(L, CAST_OFFSET(i));
        lua_pushstring(L, dataset_info[i].getCastDatatype());
        lua_settable(L, LUA_REGISTRYINDEX);

        /* Size */
        lua_pushlightuserdata(L, SIZE_OFFSET(i));
        lua_pushnumber(L, dataset_info[i].getStorageSize());
        lua_settable(L, LUA_REGISTRYINDEX);
    }

    int retValue = luaL_loadbuffer(L, bytecode, bytecode_size, "hdf5_udf_bytecode");
    if (retValue != 0)
    {
        fprintf(stderr, "luaL_loadbuffer failed: %s\n", lua_tostring(L, -1));
        lua_close(L);
        return false;
    }
    if (lua_pcall(L, 0, 0 , 0) != 0)
    {
        fprintf(stderr, "Failed to load the bytecode: %s\n", lua_tostring(L, -1));
        lua_close(L);
        return false;
    }

    // Execute the user-defined-function under a separate process so that
    // seccomp can kill it (if needed) without crashing the entire program.
    //
    // Support for Windows is still experimental; there is no sandboxing as of
    // yet, and the OS doesn't provide a fork()-like API with similar semantics.
    // In that case we just let the UDF run in the same process space as the parent.
    // Note that we define fork() as a no-op that returns 0 so we can reduce the
    // amount of #ifdef blocks in the body of this function.
    bool ret = false;
    pid_t pid = fork();
    if (pid == 0)
    {
        bool ready = true;
#ifdef ENABLE_SANDBOX
        if (rules.contains("sandbox") && rules["sandbox"].get<bool>() == true)
            ready = os::initChildSandbox(libpath, rules);
#endif
        if (ready)
        {
            // Initialize the UDF library
            lua_getglobal(L, "init");
            lua_pushstring(L, libpath.c_str());
            if (lua_pcall(L, 1, 0, 0) != 0)
            {
                fprintf(stderr, "Failed to invoke the init callback: %s\n", lua_tostring(L, -1));
                lua_close(L);
                if (os::isWindows()) { return false; } else { _exit(1); }
            }

            // Call the UDF entry point
            lua_getglobal(L, "dynamic_dataset");
            if (lua_pcall(L, 0, 0, 0) != 0)
            {
                fprintf(stderr, "Failed to invoke the dynamic_dataset callback: %s\n", lua_tostring(L, -1));
                lua_close(L);
                if (os::isWindows()) { return false; } else { _exit(1); }
            }

            // Flush stdout buffer so we don't miss any messages echoed by the UDF
            fflush(stdout);
        }
        if (os::isWindows()) { ret = ready; } else { _exit(0); }
    }
    else if (pid > 0)
    {
        bool need_waitpid = true;
#ifdef ENABLE_SANDBOX
        if (rules.contains("sandbox") && rules["sandbox"].get<bool>() == true)
        {
            ret = os::initParentSandbox(libpath, rules, pid);
            need_waitpid = false;
        }
#endif
        if (need_waitpid)
        {
            int status;
            waitpid(pid, &status, 0);
            ret = WIFEXITED(status) ? WEXITSTATUS(status) == 0 : false;
        }

        // Update output HDF5 dataset with data from shared memory segment
        memcpy(output_dataset.data, mm.mm, room_size);
    }
    lua_close(L);

    return ret;
}

/* Scan the UDF file for references to HDF5 dataset names */
std::vector<std::string> LuaBackend::udfDatasetNames(std::string udf_file)
{
    std::string input;
    std::ifstream data(udf_file, std::ifstream::binary);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(data), {});
    input.assign(buffer.begin(), buffer.end());

    std::string line;
    bool is_comment = false;
    std::istringstream iss(input);
    std::vector<std::string> output;

    auto ltrim = [](std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
            return !std::isspace(ch);
        }));
    };

    while (std::getline(iss, line))
    {
        ltrim(line);
        if (line.find("--[=====[") != std::string::npos ||
            line.find("--[====[") != std::string::npos ||
            line.find("--[===[") != std::string::npos ||
            line.find("--[==[") != std::string::npos ||
            line.find("--[=[") != std::string::npos ||
            line.find("--[[") != std::string::npos)
            is_comment = true;
        else if (is_comment && (
            line.find("]=====]") != std::string::npos ||
            line.find("]====]") != std::string::npos ||
            line.find("]===]") != std::string::npos ||
            line.find("]==]") != std::string::npos ||
            line.find("]=]") != std::string::npos ||
            line.find("]]") != std::string::npos))
            is_comment = false;
        else if (! is_comment)
        {
            auto n = line.find("lib.getData");
            auto c = line.find("--");
            if (n != std::string::npos && (c == std::string::npos || c > n))
            {
                auto start = line.substr(n).find_first_of("\"");
                auto end = line.substr(n+start+1).find_first_of("\"");
                auto name = line.substr(n).substr(start+1, end);
                output.push_back(name);
            }
        }
    }
    return output;
}

// Create a textual declaration of a struct given a compound map
std::string LuaBackend::compoundToStruct(const DatasetInfo &info, bool hardcoded_name)
{
    // Lua's FFI cdef() requires the use of #pragma pack(1) to align
    // structures at a byte boundary. Packing is needed so that UDFs
    // can iterate over the binary data retrieved by H5Dread() with
    // just a struct pointer.
    std::string cstruct = "#pragma pack(1)\n";
    cstruct += "struct " + sanitizedName(info.name) + "_t {\n";
    ssize_t current_offset = 0, pad = 0;
    for (auto &member: info.members)
    {
        if (member.offset > current_offset)
        {
            auto size = member.offset - current_offset;
            cstruct += "  char _pad" + std::to_string(pad++) +"["+ std::to_string(size) +"];\n";
        }
        current_offset = member.offset + member.size;
        cstruct += "  " + member.type + " " + (hardcoded_name ? "value" : sanitizedName(member.name));
        if (member.is_char_array)
            cstruct += "[" + std::to_string(member.size) + "]";
        cstruct += ";\n";
    }
    cstruct += "};\n";
    return cstruct;
}
