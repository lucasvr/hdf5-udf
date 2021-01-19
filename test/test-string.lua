-- hdf5-udf <file.h5> test-string.lua Temperature.lua:1000:double
function dynamic_dataset()
    local input_data = lib.getData("Dataset1")
    local input_dims = lib.getDims("Dataset1")
    local udf_data = lib.getData("Temperature.lua")
    local udf_dims = lib.getDims("Temperature.lua")

    for i=1, input_dims[1] do
        print(lib.string(input_data[i]))
    end

    for i=1, udf_dims[1] do
        udf_data[i] = (i-1) * 1.0
    end
end