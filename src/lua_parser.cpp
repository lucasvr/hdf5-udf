/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: lua_parser.cpp
 *
 * Simple Lua source code scanner. We extract arguments provided on
 * calls to lib.getData() and store them in LuaParser's names[] list.
 */
#include <fstream>
#include <sstream>
#include <algorithm>
#include "lua_parser.h"

static inline void ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}

std::vector<std::string> LuaParser::parseNames()
{
    parseFile(this->fname);
    return this->names;
}

bool LuaParser::parseFile(std::string fname)
{
    std::string input;
    std::ifstream data(fname, std::ifstream::binary);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(data), {});
    input.assign(buffer.begin(), buffer.end());
    return parseBuffer(input);
}

bool LuaParser::parseBuffer(std::string buffer)
{
    std::istringstream iss(buffer);
    std::string line;
    bool is_comment = false;
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
                this->names.push_back(name);
            }
        }
    }
    return true;
}
