/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: lua_parser.h
 *
 * Interfaces of our simple Lua source code scanner.
 */
#ifndef __lua_parser_h
#define __lua_parser_h

#include <vector>
#include <string>

class LuaParser {
public:
    LuaParser(std::string path) : fname(path) {}
    std::vector<std::string> parseNames();

private:
    bool parseFile(std::string fname);
    bool parseBuffer(std::string buffer);

    std::string fname;
    std::vector<std::string> names;
};

#endif /* __lua_parser_h */
