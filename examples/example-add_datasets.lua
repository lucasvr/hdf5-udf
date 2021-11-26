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
    local a = lib.getData("Dataset1")
    local b = lib.getData("Dataset2")
    local udf_data = lib.getData("UserDefinedDataset-Lua")
    local udf_dims = lib.getDims("UserDefinedDataset-Lua")
    local math = require("math")

    -- A gentle reminder that indexes in Lua start at 1
    local N = udf_dims[1] * udf_dims[2]

    for i=1, N do
        n2 = (2*a[i]+1) * (2*a[i]+1)
        udf_data[i] = (2*a[i]+1 - math.sqrt(n2-8*(a[i]-b[i])))/2
    end
end
