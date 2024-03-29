/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: backend_python.cpp
 *
 * Python code parser and bytecode generation/execution.
 */
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <errno.h>
#include <fstream>
#include <sstream>
#include <string>
#include <locale>
#include <codecvt>
#include <algorithm>
#include "config.h"
#include "udf_template_py.h"
#include "backend_python.h"
#include "backend_cpp.h"
#include "file_search.h"
#include "anon_mmap.h"
#include "dataset.h"
#include "os.h"

// Dataset names, sizes, and types
static std::vector<DatasetInfo> dataset_info;

// Path to the input HDF5 file
static std::string input_hdf5_path;

// Buffer used to hold the compound-to-struct name produced by pythonGetCast()
static char compound_cast_name[256];

/* Functions exported to the Python template library */
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
            return dataset_info[i].getDatatypeName();
    fprintf(stderr, "%s: dataset %s not found\n", __func__, element);
    return NULL;
}

extern "C" const char *pythonGetCast(const char *element)
{
    for (size_t i=0; i<dataset_info.size(); ++i) {
        if (dataset_info[i].name.compare(element) == 0)
        {
            auto cast = dataset_info[i].getCastDatatype();
            if (! strcmp(cast, "void*") || ! strcmp(cast, "char*"))
            {
                // Cast compound structure or the structure that supports
                // the declaration of string datasets
                PythonBackend backend;
                memset(compound_cast_name, 0, sizeof(compound_cast_name));
                snprintf(compound_cast_name, sizeof(compound_cast_name)-1,
                    "struct %s_t *", backend.sanitizedName(element).c_str());
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

extern "C" const char *pythonGetFilePath()
{
    return input_hdf5_path.c_str();
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
    std::string compound_declarations,
    std::string &source_code,
    std::vector<DatasetInfo> &datasets)
{
    AssembleData data = {
        .udf_file                 = udf_file,
        .template_string          = std::string((char *) udf_template_py),
        .compound_placeholder     = "// compound_declarations_placeholder",
        .compound_decl            = compound_declarations,
        .methods_decl_placeholder = "",
        .methods_decl             = "",
        .methods_impl_placeholder = "",
        .methods_impl             = "",
        .callback_placeholder     = "# user_callback_placeholder",
        .extension                = this->extension()
    };
    auto py_file = Backend::assembleUDF(data);
    if (py_file.size() == 0)
    {
        fprintf(stderr, "Will not be able to compile the UDF code\n");
        return "";
    }

    char *cmd[] = {
        (char *) "python3",
        (char *) "-m",
        (char *) "compileall",
        (char *) "-q",         // output error messages only
        (char *) "-l",         // don't recurse into subdirectories
        (char *) "-f",         // force rebuild even if timestamps are up to date
        (char *) py_file.c_str(),
        (char *) NULL
    };
    if (os::execCommand(cmd[0], cmd, NULL) == false)
    {
        fprintf(stderr, "Failed to build the UDF\n");
        unlink(py_file.c_str());
        return "";
    }

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

    // Find the generated bytecode file
    std::vector<std::string> results;
    std::stringstream pycache, pattern;
    pycache << parentdir << "/__pycache__";
    pattern << filename << ".cpython-*.pyc";
    bool ret = findByPattern(pycache.str(), pattern.str(), results);
    if (ret == false || results.size() == 0)
    {
        fprintf(stderr, "No bytecode found under %s\n", pattern.str().c_str());
        unlink(py_file.c_str());
        rmdir(pycache.str().c_str());
        return "";
    }

    // Assume the very first match is the one we're looking for
    std::string pyc_file = results[0];

    // Read generated bytecode
    std::string bytecode;
    std::ifstream datastream(pyc_file, std::ifstream::binary);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(datastream), {});
    bytecode.assign(buffer.begin(), buffer.end());
    datastream.close();

    // Read source file
    std::ifstream ifs(py_file.c_str());
    source_code = std::string((std::istreambuf_iterator<char>(ifs)),
        (std::istreambuf_iterator<char>()));
    ifs.close();

    unlink(py_file.c_str());
    unlink(pyc_file.c_str());
    rmdir(pycache.str().c_str());
    return bytecode;
}

/* Helper class: manage Python interpreter lifecycle */
struct PythonInterpreter {
    PythonInterpreter() :
        libpython(NULL),
        nested_session(false),
        interpreter(NULL),
        my_state(NULL)
    {
    }

    bool init() {
        // We currently depend on Python 3
        if (PY_MAJOR_VERSION != 3)
        {
            fprintf(stderr, "Error: Python3 is required\n");
            return false;
        }

        nested_session = Py_IsInitialized();
        if (nested_session)
        {
            // Create a new thread state object and make it current
            interpreter = PyInterpreterState_Main();
            my_state = PyThreadState_New(interpreter);
            PyEval_AcquireThread(my_state);
        }
        else
        {
            // Initialize the interpreter
            Py_InitializeEx(0);

            // Workaround for CFFI import errors due to missing symbols. We force libpython
            // to be loaded and for all symbols to be resolved by dlopen()
            std::string libpython_path = os::sharedLibraryName("python3");
            void *libpython = dlopen(libpython_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if (! libpython)
            {
                char libname[64];
                snprintf(libname, sizeof(libname)-1, "python3.%d", PY_MINOR_VERSION);
                libpython_path = os::sharedLibraryName(libname);
                libpython = dlopen(libpython_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
                if (! libpython)
                    fprintf(stderr, "Warning: could not load %s\n", libpython_path.c_str());
            }
        }
        return true;
    }

    ~PythonInterpreter() {
        PyErr_Clear();
        for (auto mod = decref.rbegin(); mod != decref.rend(); ++mod)
            Py_XDECREF(*mod);
        if (nested_session)
        {
            PyEval_ReleaseThread(my_state);
            PyThreadState_Clear(my_state);
            PyThreadState_Delete(my_state);
        }
        else
            Py_Finalize();
        if (libpython)
            dlclose(libpython);
        dataset_info.clear();
    }

    // dlopen handle
    void *libpython;

    // Have we been invoked from an existing Python session?
    bool nested_session;

    // List of objects we have to Py_DECREF() on exit
    std::vector<PyObject *> decref;

    // Python thread state
    PyInterpreterState *interpreter;
    PyThreadState *my_state;
};

/* Execute the user-defined-function embedded in the given buffer */
bool PythonBackend::run(
    const std::string libpath,
    const std::vector<DatasetInfo> &input_datasets,
    const DatasetInfo &output_dataset,
    const char *output_cast_datatype,
    const char *bytecode,
    size_t bytecode_size,
    const json &rules)
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
    if (! mm.createMapFor(output_dataset.data))
        return false;

    // Let output_dataset.data point to the shared memory segment
    DatasetInfo output_dataset_copy(output_dataset);
    output_dataset_copy.data = mm.mm;

    // Populate global vector of dataset names, sizes, and types
    dataset_info.push_back(std::move(output_dataset_copy));
    dataset_info.insert(
        dataset_info.end(), input_datasets.begin(), input_datasets.end());

    input_hdf5_path = this->hdf5_file_path;

    // Init Python interpreter if needed
    PythonInterpreter python;
    if (python.init() == false)
        return false;

    // The offset to the actual bytecode may change across Python versions.
    // Please look under $python_sources/Lib/importlib/_bootstrap_external.py
    // for the implementation of _validate_*() so you can identify the right
    // offset for a version of Python not featured in the list below.
    size_t bytecode_start;
    switch (PY_MINOR_VERSION)
    {
        case 1 ... 2:
            bytecode_start = 8;
            break;
        case 3 ... 6:
            bytecode_start = 12;
            break;
        case 7 ... 8:
        default:
            bytecode_start = 16;
            break;
    }
    bytecode = &bytecode[bytecode_start];
    bytecode_size -= bytecode_start;

    // Get a reference to the code object we compiled before
    PyObject *obj = PyMarshal_ReadObjectFromString(bytecode, bytecode_size);
    if (! obj)
    {
        PyObject *err = PyErr_Occurred();
        if (err)
            PyErr_Print();
        return false;
    }
    python.decref.push_back(obj);

    PyObject *module = PyImport_ExecCodeModule("udf_module", obj);
    if (! module)
    {
        fprintf(stderr, "Failed to import code object\n");
        PyErr_Print();
        return false;
    }
    python.decref.push_back(module);

    // Load essential modules prior to the launch of the user-defined-function
    // so we can keep strict sandbox rules for third-party code.
    PyObject *module_name = PyUnicode_FromString("cffi");
    python.decref.push_back(module_name);
    PyObject *cffi_module = PyImport_Import(module_name);
    if (! cffi_module)
    {
        fprintf(stderr, "Failed to import the cffi module\n");
        return false;
    }
    python.decref.push_back(cffi_module);

    // From the documentation: unlike other functions that steal references,
    // PyModule_AddObject() only decrements the reference count of value on success
    Py_INCREF(cffi_module);
    if (PyModule_AddObject(module, "cffi", cffi_module) < 0)
    {
        Py_DECREF(cffi_module);
        return false;
    }

    // Construct an instance of cffi.FFI()
    PyObject *cffi_dict = PyModule_GetDict(cffi_module);
    PyObject *ffi = cffi_dict ? PyDict_GetItemString(cffi_dict, "FFI") : NULL;
    PyObject *ffi_instance = ffi ? PyObject_CallObject(ffi, NULL) : NULL;
    python.decref.push_back(ffi_instance);

    // Get a reference to cffi.FFI().dlopen()
    PyObject *ffi_dlopen = ffi_instance ? PyObject_GetAttrString(ffi_instance, "dlopen") : NULL;
    if (! ffi_dlopen)
    {
        fprintf(stderr, "Failed to retrieve method cffi.FFI().dlopen()\n");
        return false;
    }
    python.decref.push_back(ffi_dlopen);

    // Get handles for lib.load() and for the dynamic_dataset() UDF entry point
    bool retval = false;
    PyObject *dict = PyModule_GetDict(module);
    PyObject *lib = dict ? PyDict_GetItemString(dict, "lib") : NULL;
    PyObject *loadlib = lib ? PyObject_GetAttrString(lib, "load") : NULL;
    PyObject *udf = dict ? PyDict_GetItemString(dict, "dynamic_dataset") : NULL;

    if (! lib)
        fprintf(stderr, "Failed to find symbol 'lib' in code object\n");
    else if (! loadlib)
        fprintf(stderr, "Failed to find symbol 'lib.load()' in code object\n");
    else if (! udf)
        fprintf(stderr, "Failed to find entry point 'dynamic_dataset()' in code object\n");
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
            return false;
        }

        // lib.udflib = lib.ffi.dlopen(libpath)
        PyObject *pyargs = PyTuple_New(1);
        PyObject *pypath = Py_BuildValue("s", libpath.c_str());
        PyTuple_SetItem(pyargs, 0, pypath);
        PyObject *dlopen_ret = PyObject_CallObject(ffi_dlopen, pyargs);
        python.decref.push_back(pyargs);
        if (dlopen_ret)
        {
            python.decref.push_back(dlopen_ret);
            if (PyObject_SetAttrString(lib, "udflib", dlopen_ret) < 0)
            {
                fprintf(stderr, "Failed to initialize lib.ffi\n");
                return false;
            }
        }

        // Execute the user-defined function
        retval = executeUDF(loadlib, udf, libpath, rules);
        if (retval == true)
        {
            // Update output HDF5 dataset with data from shared memory segment
            memcpy(output_dataset.data, mm.mm, room_size);
        }
    }

    return retval;
}

/* Coordinate the execution of the UDF under a separate process */
bool PythonBackend::executeUDF(
    PyObject *loadlib,
    PyObject *udf,
    std::string libpath,
    const json &rules)
{
    /*
     * Execute the user-defined-function under a separate process so that
     * seccomp can kill it (if needed) without crashing the entire program
     *
     * Support for Windows is still experimental; there is no sandboxing as of
     * yet, and the OS doesn't provide a fork()-like API with similar semantics.
     * In that case we just let the UDF run in the same process space as the parent.
     * Note that we define fork() as a no-op that returns 0 so we can reduce the
     * amount of #ifdef blocks in the body of this function.
     */
    bool retval = false;
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
            // Run 'lib.load(libpath)' from udf_template_py
            PyObject *pyargs = PyTuple_New(1);
            PyObject *pypath = Py_BuildValue("s", libpath.c_str());
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
        if (os::isWindows()) { retval = ready; } else { _exit(ready ? 0  : 1); }
    }
    else if (pid > 0)
    {
        bool need_waitpid = true;
#ifdef ENABLE_SANDBOX
        if (rules.contains("sandbox") && rules["sandbox"].get<bool>() == true)
        {
            retval = os::initParentSandbox(libpath, rules, pid);
            need_waitpid = false;
        }
#endif
        if (need_waitpid)
        {
            int status;
            waitpid(pid, &status, 0);
            retval = WIFEXITED(status) ? WEXITSTATUS(status) == 0 : false;
        }
    }

    return retval;
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
            if (start == std::string::npos && end == std::string::npos)
            {
                start = line.substr(n).find_first_of("'");
                end = line.substr(n+start+1).find_first_of("'");
            }
            auto name = line.substr(n).substr(start+1, end);
            output.push_back(name);
        }
    }
    return output;
}

// Create a textual declaration of a struct given a compound map
std::string PythonBackend::compoundToStruct(const DatasetInfo &info, bool hardcoded_name)
{
    // Python's CFFI cdef() does not recognize packing attributes
    // such as __attribute__((pack)) or #pragma pack. Rather, it
    // provides a special argument 'packed=True' that instructs
    // the parser to align all structure fields at a byte boundary.
    // Packing is needed so that UDFs can iterate over the binary
    // data retrieved by H5Dread() with just a struct pointer.
    std::string cstruct = "struct " + sanitizedName(info.name) + "_t {\n";
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
