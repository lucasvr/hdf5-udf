-- hdf5-udf <file.h5> test-compound-string.lua Temperature.lua:1000:double
function dynamic_dataset()
    local compound = lib.getData("Dataset1")
    local udf_data = lib.getData("Temperature.lua")
    local udf_dims = lib.getDims("Temperature.lua")

    for i=0, udf_dims[1]-1 do
        print(string.format(
            "serial: %d, location: %s, temperature: %.6f, pressure: %.6f",
            tonumber(compound[i].serial_number),
            lib.string(compound[i].location),
            tonumber(compound[i].temperature),
            tonumber(compound[i].pressure)))
        udf_data[i] = compound[i].temperature
    end
end