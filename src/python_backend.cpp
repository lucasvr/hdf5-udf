/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: python_backend.cpp
 *
 * Python code parser and bytecode generation/execution.
 */
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <errno.h>
#include <glob.h>
#include <fstream>
#include <sstream>
#include <string>
#include <locale>
#include <codecvt>
#include <algorithm>
#include "python_backend.h"
#include "cpp_backend.h"
#include "anon_mmap.h"
#include "dataset.h"
#ifdef ENABLE_SANDBOX
#include "sandbox.h"
#endif

// Dataset names, sizes, and types
static std::vector<DatasetInfo> dataset_info;

// Buffer used to hold the compound-to-struct name produced by pythonGetCast()
static char compound_cast_name[256];

/* Functions exported to the Python template library (udf_template.py) */
extern "C" void *pythonGetData(const char *element)
{
    for (size_t i=0; i<dataset_info.size(); ++i)
        if (dataset_info[i].name.compare(element) == 0)
            return dataset_info[i].data;
    fprintf(stderr, "%s: dataset %s not found\n", __func__, element);
    return NULL;
}

extern "C" const char *pythonGetType(const char *element)
{
    for (size_t i=0; i<dataset_info.size(); ++i)
        if (dataset_info[i].name.compare(element) == 0)
            return dataset_info[i].getDatatype();
    fprintf(stderr, "%s: dataset %s not found\n", __func__, element);
    return NULL;
}

extern "C" const char *pythonGetCast(const char *element)
{
    for (size_t i=0; i<dataset_info.size(); ++i) {
        if (dataset_info[i].name.compare(element) == 0)
        {
            auto cast = dataset_info[i].getCastDatatype();
            if (! strcmp(cast, "void*"))
            {
                // Cast compound structure
                PythonBackend backend;
                memset(compound_cast_name, 0, sizeof(compound_cast_name));
                snprintf(compound_cast_name, sizeof(compound_cast_name)-1,
                    "struct compound_%s *", backend.sanitizedName(element).c_str());
                return compound_cast_name;
            }
            return cast;
        }
    }
    fprintf(stderr, "%s: dataset %s not found\n", __func__, element);
    return NULL;
}

extern "C" const char *pythonGetDims(const char *element)
{
    for (size_t i=0; i<dataset_info.size(); ++i)
        if (dataset_info[i].name.compare(element) == 0)
            return dataset_info[i].dimensions_str.c_str();
    fprintf(stderr, "%s: dataset %s not found\n", __func__, element);
    return NULL;
}

/* This backend's name */
std::string PythonBackend::name()
{
    return "CPython";
}

/* Extension managed by this backend */
std::string PythonBackend::extension()
{
    return ".py";
}

/* Compile Python to a bytecode. Returns the bytecode as a string object. */
std::string PythonBackend::compile(
    std::string udf_file,
    std::string template_file,
    std::string compound_declarations)
{
    AssembleData data = {
        .udf_file = udf_file,
        .template_file = template_file,
        .compound_declarations = compound_declarations,
        .callback_placeholder = "# user_callback_placeholder",
        .compound_placeholder = "// compound_declarations_placeholder",
        .extension = this->extension()
    };
    auto py_file = Backend::assembleUDF(data);
    if (py_file.size() == 0)
    {
        fprintf(stderr, "Will not be able to compile the UDF code\n");
        return "";
    }

    pid_t pid = fork();
    if (pid == 0)
    {
        // Child process
        char *cmd[] = {
            (char *) "python3",
            (char *) "-m",
            (char *) "compileall",
            (char *) "-l",         // don't recurse into subdirectories
            (char *) "-f",         // force rebuild even if timestamps are up to date
            (char *) py_file.c_str(),
            (char *) NULL
        };
        execvp(cmd[0], cmd);
    }
    else if (pid > 0)
    {
        // Parent
        int exit_status;
        wait4(pid, &exit_status, 0, NULL);

        // Find the bytecode
        auto sep = py_file.find_last_of('/');
        if (sep == std::string::npos)
        {
            fprintf(stderr, "Failed to identify directory where assembled file was saved\n");
            unlink(py_file.c_str());
            return "";
        }
        std::string parentdir = py_file.substr(0, sep);
        std::string filename = py_file.substr(sep + 1);
        sep = filename.find_last_of(".");
        filename = filename.substr(0, sep);

        // Because the Python version is part of the generated file name, we resort to glob()
        // to identify the actual path to that file.
        glob_t results;
        std::stringstream pycache, pattern;
        pycache << parentdir << "/__pycache__";
        pattern << pycache.str() << "/" << filename << ".cpython-*.pyc";
        int ret = glob(pattern.str().c_str(), GLOB_NOSORT, NULL, &results);
        if (ret != 0 || results.gl_pathc == 0)
        {
            fprintf(stderr, "No bytecodes were found under %s\n", pattern.str().c_str());
            unlink(py_file.c_str());
            rmdir(pycache.str().c_str());
            if (ret == 0) { globfree(&results); }
            return "";
        }

        // Assume the very first match is the one we're looking for
        std::string pyc_file = results.gl_pathv[0];
        globfree(&results);

        // Read generated bytecode
        std::string bytecode;
        std::ifstream data(pyc_file, std::ifstream::binary);
        std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(data), {});
        bytecode.assign(buffer.begin(), buffer.end());
        
        unlink(py_file.c_str());
        unlink(pyc_file.c_str());
        rmdir(pycache.str().c_str());
        return bytecode;
    }
    fprintf(stderr, "Failed to execute python3\n");
    return "";
}

/* Helper function: deinitializes the Python interpreter */
static void teardown(std::vector<PyObject *> decref, void *libpython)
{
    for (auto mod = decref.rbegin(); mod != decref.rend(); ++mod)
        Py_XDECREF(*mod);
    Py_Finalize();
    if (libpython)
        dlclose(libpython);
}

/* Execute the user-defined-function embedded in the given buffer */
bool PythonBackend::run(
    const std::string filterpath,
    const std::vector<DatasetInfo> input_datasets,
    const DatasetInfo output_dataset,
    const char *output_cast_datatype,
    const char *bytecode,
    size_t bytecode_size)
{
    if (bytecode_size < 16)
    {
        fprintf(stderr, "Error: Python bytecode is too small to be valid\n");
        return false;
    }

    /*
     * We want to make the output dataset writeable by the UDF. Because
     * the UDF is run under a separate process we have to use a shared
     * memory segment which both processes can read and write to.
     */
    size_t room_size = output_dataset.getGridSize() * output_dataset.getStorageSize();
    AnonymousMemoryMap mm(room_size);
    if (! mm.create())
        return false;

    // Let output_dataset.data point to the shared memory segment
    DatasetInfo output_dataset_copy = output_dataset;
    output_dataset_copy.data = mm.mm;

    // Populate global vector of dataset names, sizes, and types
    dataset_info.push_back(output_dataset_copy);
    dataset_info.insert(
        dataset_info.end(), input_datasets.begin(), input_datasets.end());

    // List of objects we have to Py_DECREF() on exit
    std::vector<PyObject *> decref;

    // Workaround for CFFI import errors due to missing symbols. We force libpython
    // to be loaded and for all symbols to be resolved by dlopen()
    void *libpython = dlopen("libpython3.so", RTLD_NOW | RTLD_GLOBAL);
    if (! libpython)
    {
        char libname[64];
        snprintf(libname, sizeof(libname)-1, "libpython3.%d.so", PY_MINOR_VERSION);
        libpython = dlopen(libname, RTLD_NOW | RTLD_GLOBAL);
    }

    // Init Python interpreter
    Py_Initialize();

    // We have to check whether this offset is fixed at all times or if can
    // change. Some docs mention an offset of 8 bytes, for instance.
    bytecode = &bytecode[16];
    bytecode_size -= 16;

    // Get a reference to the code object we compiled before
    PyObject *obj = PyMarshal_ReadObjectFromString(bytecode, bytecode_size);
    if (! obj)
    {
        PyObject *err = PyErr_Occurred();
        if (err && (
            PyErr_GivenExceptionMatches(err, PyExc_EOFError) ||
            PyErr_GivenExceptionMatches(err, PyExc_ValueError) ||
            PyErr_GivenExceptionMatches(err, PyExc_TypeError)))
        {
            PyErr_Print();
        }
        PyErr_Clear();
        teardown(decref, libpython);
        return false;
    }
    decref.push_back(obj);

    PyObject *module = PyImport_ExecCodeModule("udf_module", obj);
    if (! module)
    {
        fprintf(stderr, "Failed to import code object\n");
        PyErr_Print();
        teardown(decref, libpython);
        return false;
    }
    decref.push_back(module);

    // Load essential modules prior to the launch of the user-defined-function
    // so we can keep strict sandbox rules for third-party code.
    PyObject *module_name = PyUnicode_FromString("cffi");
    decref.push_back(module_name);
    PyObject *cffi_module = PyImport_Import(module_name);
    if (! cffi_module)
    {
        fprintf(stderr, "Failed to import the cffi module\n");
        teardown(decref, libpython);
        return false;
    }
    decref.push_back(cffi_module);

    // From the documentation: unlike other functions that steal references,
    // PyModule_AddObject() only decrements the reference count of value on success
    Py_INCREF(cffi_module);
    if (PyModule_AddObject(module, "cffi", cffi_module) < 0)
    {
        Py_DECREF(cffi_module);
        teardown(decref, libpython);
        return false;
    }

    // Construct an instance of cffi.FFI()
    PyObject *cffi_dict = PyModule_GetDict(cffi_module);
    PyObject *ffi = cffi_dict ? PyDict_GetItemString(cffi_dict, "FFI") : NULL;
    PyObject *ffi_instance = ffi ? PyObject_CallObject(ffi, NULL) : NULL;
    decref.push_back(ffi_instance);

    // Get a reference to cffi.FFI().dlopen()
    PyObject *ffi_dlopen = ffi_instance ? PyObject_GetAttrString(ffi_instance, "dlopen") : NULL;
    if (! ffi_dlopen)
    {
        fprintf(stderr, "Failed to retrieve method cffi.FFI().dlopen()\n");
        teardown(decref, libpython);
        return false;
    }
    decref.push_back(ffi_dlopen);

    // Get handles for lib.load() and for the dynamic_dataset() UDF entry point
    bool retval = false;
    PyObject *dict = PyModule_GetDict(module);
    PyObject *lib = dict ? PyDict_GetItemString(dict, "lib") : NULL;
    PyObject *loadlib = lib ? PyObject_GetAttrString(lib, "load") : NULL;
    PyObject *udf = dict ? PyDict_GetItemString(dict, "dynamic_dataset") : NULL;

    if (! lib || ! loadlib || ! udf)
        fprintf(stderr, "Failed to load required symbols from code object\n");
    else if (! PyCallable_Check(loadlib))
        fprintf(stderr, "Error: lib.load is not a callable function\n");
    else if (! PyCallable_Check(udf))
        fprintf(stderr, "Error: dynamic_dataset is not a callable function\n");
    else
    {
        // lib.ffi = cffi.FFI()
        if (PyObject_SetAttrString(lib, "ffi", ffi_instance) < 0)
        {
            fprintf(stderr, "Failed to initialize lib.ffi\n");
            teardown(decref, libpython);
            return false;
        }

        // lib.filterlib = lib.ffi.dlopen(filterpath)
        PyObject *pyargs = PyTuple_New(1);
        PyObject *pypath = Py_BuildValue("s", filterpath.c_str());
        PyTuple_SetItem(pyargs, 0, pypath);
        PyObject *dlopen_ret = PyObject_CallObject(ffi_dlopen, pyargs);
        decref.push_back(pyargs);
        if (dlopen_ret)
        {
            decref.push_back(dlopen_ret);
            if (PyObject_SetAttrString(lib, "filterlib", dlopen_ret) < 0)
            {
                fprintf(stderr, "Failed to initialize lib.ffi\n");
                teardown(decref, libpython);
                return false;
            }
        }

        // Execute the user-defined function
        retval = executeUDF(loadlib, udf, filterpath);
        if (retval == true)
        {
            // Update output HDF5 dataset with data from shared memory segment
            memcpy(output_dataset.data, mm.mm, room_size);
        }
    }

    teardown(decref, libpython);
    return retval;
}

/* Coordinate the execution of the UDF under a separate process */
bool PythonBackend::executeUDF(PyObject *loadlib, PyObject *udf, std::string filterpath)
{
    /*
     * Execute the user-defined-function under a separate process so that
     * seccomp can kill it (if needed) without crashing the entire program
     */
    pid_t pid = fork();
    if (pid == 0)
    {
        bool ready = true;
#ifdef ENABLE_SANDBOX
        Sandbox sandbox;
        auto paths_allowed = pathsAllowed();
        ready = sandbox.init(filterpath, paths_allowed);
#endif
        if (ready)
        {
            // Run 'lib.load(filterpath)' from our udf_template.py
            PyObject *pyargs = PyTuple_New(1);
            PyObject *pypath = Py_BuildValue("s", filterpath.c_str());
            PyTuple_SetItem(pyargs, 0, pypath);
            PyObject *loadret = PyObject_CallObject(loadlib, pyargs);

            // Run 'dynamic_dataset()' defined by the user
            PyObject *callret = PyObject_CallObject(udf, NULL);
            if (! callret)
            {
                // Function call terminated by an exception
                PyErr_Print();
                PyErr_Clear();
                ready = false;
            }
            Py_XDECREF(loadret);
            Py_XDECREF(callret);
            Py_XDECREF(pyargs);

            // Flush stdout buffer so we don't miss any messages echoed by the UDF
            fflush(stdout);
        }
        // Exit the process without invoking any callbacks registered with atexit()
        _exit(ready ? 0 : 1);
    }
    else if (pid > 0)
    {
        int status;
        waitpid(pid, &status, 0);
        return WIFEXITED(status) ? WEXITSTATUS(status) == 0 : false;
    }
    return false;
}

/* List of paths we need to access (called after Py_Initialize()) */
std::vector<std::string> PythonBackend::pathsAllowed()
{
    std::vector<std::string> paths_allowed;
    std::wstringstream input(Py_GetPath());
    std::wstring path;

    // wstring -> string converter
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;

    while (std::getline(input, path, L':'))
    {
        std::stringstream ss1, ss2, ss3;
        ss1 << converter.to_bytes(path) << "/pycparser";
        ss2 << converter.to_bytes(path) << "/pycparser/*";
        ss3 << converter.to_bytes(path) << "/pycparser/__pycache__/*";
        paths_allowed.push_back(ss1.str());
        paths_allowed.push_back(ss2.str());
        paths_allowed.push_back(ss3.str());
    }
    return paths_allowed;
}

/* Debug helper */
void PythonBackend::printPyObject(PyObject *obj)
{
    PyObject *repr = PyObject_Repr(obj);
    const char *s = PyUnicode_AsUTF8(repr);
    printf("%p=%s\n", repr, s);
}

/* Scan the UDF file for references to HDF5 dataset names */
std::vector<std::string> PythonBackend::udfDatasetNames(std::string udf_file)
{
    std::string input;
    std::ifstream data(udf_file, std::ifstream::binary);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(data), {});
    input.assign(buffer.begin(), buffer.end());

    std::string line;
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
        auto n = line.find("lib.getData");
        auto c = line.find("#");
        if (n != std::string::npos && (c == std::string::npos || c > n))
        {
            auto start = line.substr(n).find_first_of("\"");
            auto end = line.substr(n+start+1).find_first_of("\"");
            auto name = line.substr(n).substr(start+1, end);
            output.push_back(name);
        }
    }
    return output;
}

// Create a textual declaration of a struct given a compound map
std::string PythonBackend::compoundToStruct(const DatasetInfo info)
{
    // Python's CFFI cdef() does not recognize packing attributes
    // such as __attribute__((pack)) or #pragma pack. Rather, it
    // provides a special argument 'packed=True' that instructs
    // the parser to align all structure fields at a byte boundary.
    // Packing is needed so that UDFs can iterate over the binary
    // data retrieved by H5Dread() with just a struct pointer.
    std::string cstruct = "struct compound_" + sanitizedName(info.name) + " {\n";
    size_t current_offset = 0, pad = 0;
    for (auto &member: info.members)
    {
        if (member.offset > current_offset)
        {
            auto size = member.offset - current_offset;
            cstruct += "  char _pad" + std::to_string(pad++) +"["+ std::to_string(size) +"];\n";
        }
        current_offset = member.offset + member.size;
        cstruct += "  " + member.type + " " + sanitizedName(member.name);
        if (member.is_char_array)
            cstruct += "[" + std::to_string(member.size) + "]";
        cstruct += ";\n";
    }
    cstruct += "};\n";
    return cstruct;
}