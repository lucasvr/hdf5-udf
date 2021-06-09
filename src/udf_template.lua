--
-- HDF5-UDF: User-Defined Functions for HDF5
--
-- File: udf_template.lua
--
-- HDF5 filter callbacks and main interface with the Lua API.
--

local lib = {}

function init(libpath)
    local ffi = require("ffi")
    local udflib = ffi.load(libpath)
    ffi.cdef[[
        void       *luaGetData(const char *);
        const char *luaGetType(const char *);
        const char *luaGetCast(const char *);
        const char *luaGetDims(const char *);
        int         luaGetElementSize(const char *);
        // compound_declarations_placeholder
    ]]

    lib.string = function(name)
        local type = tostring(ffi.typeof(name)):gsub("ctype<", ""):gsub("( ?)[&>?]", "")
        if type:find("^struct") ~= nil then
            return ffi.string(name.value)
        elseif type:find("^char ") ~= nil then
            return ffi.string(ffi.cast("char *", name))
        end
        return ffi.string(name)
    end

    lib.setString = function(name, s)
        local t = tostring(ffi.typeof(name)):gsub("ctype<", ""):gsub("( ?)[&>?]", "")
        if t:find("^struct") ~= nil then
            local n = #s
            if n > ffi.sizeof(name.value) then
                n = ffi.sizeof(name.value)
            end
            ffi.copy(name.value, s, n)
        else
            local n = #s
            if n > ffi.sizeof(name) then
                n = ffi.sizeof(name)
            end
            ffi.copy(name, s, n)
        end
    end

    lib.getData = function(name)
        local cast = udflib.luaGetCast(name)
        local data = ffi.cast("char*", udflib.luaGetData(name))

        -- To allow 1-based indexing of HDF5 datasets we return a shifted data
        -- container to the Lua application. In case of compounds or structures
        -- the shift size is determined by ffi.sizeof().
        local elementsize = udflib.luaGetElementSize(name)
        if elementsize == -1 then
            local datatype = ffi.string(cast):gsub("*", "")
            elementsize = ffi.sizeof(ffi.typeof(datatype))

        end
        return ffi.cast(ffi.string(cast), data - elementsize)
    end

    lib.getType = function(name)
        return ffi.string(udflib.luaGetType(name))
    end

    lib.getDims = function(name)
        local dims = ffi.string(udflib.luaGetDims(name))
        local t = {}
        for dim in string.gmatch(dims, "([^x]+)") do
            table.insert(t, tonumber(dim))
        end
        return t
    end
end

-- User-Defined Function

-- user_callback_placeholder
