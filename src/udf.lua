--
-- HDF5-UDF: User-Defined Functions for HDF5
--
-- File: udf.lua
--
-- HDF5 filter callbacks and main interface with the Lua API.
--

local lib = {}

function init(filterpath)
    local ffi = require("ffi")
    local filterlib = ffi.load(filterpath)
    ffi.cdef[[
        void       *get_data(const char *);
        const char *get_type(const char *);
        const char *get_cast(const char *);
        const char *get_dims(const char *);
    ]]

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
end

-- User-Defined Function

-- user_callback_placeholder