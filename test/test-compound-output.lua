--
-- Shows how to generate a compound dataset from a UDF
--
-- To embed this UDF in an existing HDF5 file, run:
-- $ make files
-- $ hdf5-udf compound.h5 test-compound-output.lua 'Observations.lua:{id:uint32,location:string,temperature:float}:1000'
--

function dynamic_dataset()
    local udf_data = lib.getData("Observations.lua")
    local udf_dims = lib.getDims("Observations.lua")

    for i=1, udf_dims[1] do
        udf_data[i].id = i-1
        udf_data[i].location = "location_" .. i-1
        udf_data[i].temperature = 100.0 + i-1
    end
end