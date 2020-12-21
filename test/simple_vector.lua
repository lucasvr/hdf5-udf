-- hdf5-udf simple_vector.h5 simple_vector.lua Simple.lua:1500:float
function dynamic_dataset()
    local udf_data = lib.getData("Simple.lua")
    local udf_dims = lib.getDims("Simple.lua")
    local N = udf_dims[1]
    for i=0, N-1 do
        udf_data[i] = i
    end
end
