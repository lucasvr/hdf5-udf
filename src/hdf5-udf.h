/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: hdf5-udf.h
 *
 * Public C/C++ API
 */
#ifndef __hdf5_udf_h
#define __hdf5_udf_h

extern "C" {

#include <stdbool.h>

// Forward declaration of libudf's opaque context data structure
typedef struct udf_context udf_context;

// Initialize the library.
// Returns a pointer to a dynamically allocated context on success or NULL on failure.
// The context must be freed by passing it to libudf_destroy().
udf_context *libudf_init(const char *hdf5_file, const char *udf_file);

// Release any resources allocated by the library.
void libudf_destroy(udf_context *ctx);

// Set a key/value pair.
//
// Recognized options and their corresponding valid values are:
// - "overwrite": "true", "false"
//   Configure to "true" if you wish to overwrite an existing UDF dataset
//   when storing the resulting bytecode into the HDF5 file. Defaults to
//   "false".
//
// - "save_sourcecode": "true", "false"
//   Configure to "true" if you wish to append the source code as UDF
//   metadata. Defaults to "false".
bool libudf_set_option(const char *option, const char *value, udf_context *ctx);

// Add a UDF dataset description to the compilation chain.
bool libudf_push_dataset(const char *description, udf_context *ctx);

// Compile the given UDF into bytecode/executable form.
bool libudf_compile(udf_context *ctx);

// Store the compiled UDF on the target file.
// If "metadata" is non-NULL, then up to "size" bytes of the serialized JSON
// metadata, stored in the HDF5, are returned (including the NULL terminator).
bool libudf_store(char *metadata, size_t size, udf_context *ctx);

// Get the last error message.
size_t libudf_get_error(char *buf, size_t size, udf_context *ctx);

}
#endif