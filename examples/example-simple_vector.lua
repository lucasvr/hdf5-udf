--
-- Simple example: fills-in the dataset with sequential values starting from 0
--
-- To embed it in an existing HDF5 file, run:
-- $ make files
-- $ hdf5-udf example-simple_vector.h5 example-simple_vector.lua Simple:500:float

function dynamic_dataset()
    local udf_data = lib.getData("Simple")
    local udf_dims = lib.getDims("Simple")

    -- A gentle reminder that indexes in Lua begin with 1. This is why
    -- udf_dims is indexed from 1 onwards.
    local N = udf_dims[1]

    for i=0, N-1 do
        -- Arrays retrieved from FFI through lib.getData() are allocated and
        -- managed by C code, thus indexing begins with 0 on such arrays.
        udf_data[i] = i
    end
end
