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
end
