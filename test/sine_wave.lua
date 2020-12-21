-- hdf5-udf sine_wave.h5 sine_wave.lua SineWave.lua:100x10:int32
function dynamic_dataset()
    local udf_data = lib.getData("SineWave.lua")
    local udf_type = lib.getType("SineWave.lua")
    local udf_dims = lib.getDims("SineWave.lua")

    local N = udf_dims[1]
    local M = udf_dims[2]

    for i=0, N-1 do
        for j=0, M-1 do
            udf_data[i*M + j] = math.sin(i*M + j) * 100.0
        end
    end
end
