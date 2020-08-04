--
-- Simple example: combines data from two existing datasets
--
-- To embed it in an existing HDF5 file, run:
-- $ make files
-- $ hdf5-udf example-add_datasets.h5 example-add_datasets.lua
--
-- Note the absence of an output Dataset name in the call to
-- hdf5-udf: the tool determines it based on the calls to
-- lib.getData() made by this script. The resolution and
-- data types are determined to be the same as that of the
-- input datasets, Dataset1 and Dataset2.

function dynamic_dataset()
    local ds1_data = lib.getData("Dataset1")
    local ds2_data = lib.getData("Dataset2")
    local udf_data = lib.getData("VirtualDataset")
    local udf_dims = lib.getDims("VirtualDataset")

    -- A gentle reminder that indexes in Lua begin with 1. This is why
    -- udf_dims is indexed from 1 onwards.
    local N = udf_dims[1] * udf_dims[2]

    for i=0, N-1 do
        -- Arrays retrieved from FFI through lib.getData() are allocated and
        -- managed by C code, thus indexing begins with 0 on such arrays.
        udf_data[i] = ds1_data[i] + ds2_data[i]
    end
end
