--
-- HDF5-UDF: User-Defined Functions for HDF5
--
-- File: udf_template.lua
--
-- HDF5 filter callbacks and main interface with the Lua API.
--

local lib = {}

function init(filterpath)
    local ffi = require("ffi")
    local filterlib = ffi.load(filterpath)
    ffi.cdef[[
        void       *luaGetData(const char *);
        const char *luaGetType(const char *);
        const char *luaGetCast(const char *);
        const char *luaGetDims(const char *);
        // compound_declarations_placeholder
    ]]

    lib.getData = function(name)
        return ffi.cast(ffi.string(filterlib.luaGetCast(name)), filterlib.luaGetData(name))
    end

    lib.getType = function(name)
        return ffi.string(filterlib.luaGetType(name))
    end

    lib.getDims = function(name)
        local dims = ffi.string(filterlib.luaGetDims(name))
        local t = {}
        for dim in string.gmatch(dims, "([^x]+)") do
            table.insert(t, dim)
        end
        return t
    end
end

-- User-Defined Function

-- user_callback_placeholder