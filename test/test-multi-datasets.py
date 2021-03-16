# hdf5-udf add_datasets.h5 test-multi-datasets.py
def dynamic_dataset():
    ds1_data = lib.getData("Dataset1")
    ds2_data = lib.getData("Dataset2")
    udf_data = lib.getData("UserDefinedDataset.py")
    udf_dims = lib.getDims("UserDefinedDataset.py")

    for i in range(udf_dims[0] * udf_dims[1]):
        udf_data[i] = ds1_data[i] + ds2_data[i]
