--
-- Simple example: fills-in the dataset with a sine wave.
--
-- To embed it in an existing HDF5 file, run:
-- $ make files
-- $ hdf5-udf example-sine_wave.h5 example-sine_wave.lua SineWave:100x10:int32
--
function dynamic_dataset()
    local udf_data = lib.getData("SineWave")
    local udf_type = lib.getType("SineWave")
    local udf_dims = lib.getDims("SineWave")

    -- A gentle reminder that indexes in Lua start at 1
    local N = udf_dims[1]
    local M = udf_dims[2]

    -- It is possible to generate different values for the output grid
    -- depending on the declared dataset type, as we do below. In this
    -- example, the sine wave is shifted by 100 on unsigned data types.

    if udf_type == "uint16" or udf_type == "uint32" or udf_type == "uint64" then
        for i=1, N do
            for j=1, M do
                udf_data[i*M + j] = math.sin(i*M + j) * 100 + 100
            end
        end
    else
        for i=1, N do
            for j=1, M do
                udf_data[i*M + j] = math.sin(i*M + j) * 100
            end
        end
    end
end
