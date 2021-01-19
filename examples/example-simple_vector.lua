--
-- Simple example: fills-in the dataset with sequential values starting from 0
--
-- To embed it in an existing HDF5 file, run:
-- $ make files
-- $ hdf5-udf example-simple_vector.h5 example-simple_vector.lua Simple:500:float

function dynamic_dataset()
    local udf_data = lib.getData("Simple")
    local udf_dims = lib.getDims("Simple")

    -- A gentle reminder that indexes in Lua start at 1
    for i=1, udf_dims[1] do
        udf_data[i] = i
    end
end
