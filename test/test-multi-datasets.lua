-- hdf5-udf add_datasets.h5 test-multi-datasets.lua
function dynamic_dataset()
    local ds1_data = lib.getData("Dataset1")
    local ds2_data = lib.getData("Dataset2")
    local udf_data = lib.getData("VirtualDataset.lua")
    local udf_dims = lib.getDims("VirtualDataset.lua")

    for i=1, udf_dims[1] * udf_dims[2] do
        udf_data[i] = ds1_data[i] + ds2_data[i]
    end
end