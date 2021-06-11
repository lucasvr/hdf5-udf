/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: file_search.h
 *
 * Very simple glob()-like interfaces to search for files given a directory
 * path and file name patterns.
 */

#ifndef __file_search_h
#define __file_search_h

#include <stdbool.h>
#include <vector>
#include <string>

bool findByExtension(std::string dir, std::string ext, std::vector<std::string> &out, bool subdirs=true);
bool findByPattern(std::string dir, std::string pattern, std::vector<std::string> &out, bool subdirs=true);

#endif
