--
-- HDF5-UDF: User-Defined Functions for HDF5
--
-- File: udf.lua
--
-- HDF5 filter callbacks and main interface with the Lua API.
--

local os = require("os")
local filterpath = ""
local filterdirs = os.getenv("HDF5_PLUGIN_PATH") or "/usr/local/hdf5/lib/plugin"
for path in string.gmatch(filterdirs, "([^:]+)") do
    local candidate = path .. "/libhdf5-udf.so"
    if os.rename(candidate, candidate) then
        filterpath = candidate
        break
    end
end

local ffi = require("ffi")
local filterlib = ffi.load(filterpath)
ffi.cdef[[
    void       *get_data(const char *);
    const char *get_type(const char *);
    const char *get_cast(const char *);
    const char *get_dims(const char *);
]]

local lib = {}

lib.getData = function(name)
    return ffi.cast(ffi.string(filterlib.get_cast(name)), filterlib.get_data(name))
end

lib.getType = function(name)
    return ffi.string(filterlib.get_type(name))
end

lib.getDims = function(name)
    local dims = ffi.string(filterlib.get_dims(name))
    local t = {}
    for dim in string.gmatch(dims, "([^x]+)") do
        table.insert(t, dim)
    end
    return t
end

-- User-Defined Function

-- user_callback_placeholder