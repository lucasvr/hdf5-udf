-- hdf5-udf <file.h5> test-filepath.lua Temperature.lua:1000:double
function dynamic_dataset()
    local path = lib.getFilePath():match("^.+/(.+)$")
    print("HDF5 file path is '" .. path .. "'")
end
