-- hdf5-udf simple_vector.h5 simple_vector.lua Simple.lua:1500:float
function dynamic_dataset()
    local udf_data = lib.getData("Simple.lua")
    local udf_dims = lib.getDims("Simple.lua")
    print("udf_dims=" .. udf_dims[1])
    local N = udf_dims[1]
    for i=1, N do
        udf_data[i] = i-1
    end
end
