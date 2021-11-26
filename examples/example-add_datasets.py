#
# Simple example: combines data from two existing datasets
#
# To embed it in an existing HDF5 file, run:
# $ make files
# $ hdf5-udf example-add_datasets.h5 example-add_datasets.py
#
# Note the absence of an output Dataset name in the call to
# hdf5-udf: the tool determines it based on the calls to
# lib.getData() made by this code. The resolution and
# data types are determined to be the same as that of the
# input datasets, Dataset1 and Dataset2.
#

def dynamic_dataset():
    import math

    a = lib.getData("Dataset1")
    b = lib.getData("Dataset2")
    udf_data = lib.getData("UserDefinedDataset-Py")
    udf_dims = lib.getDims("UserDefinedDataset-Py")

    for i in range(udf_dims[0] * udf_dims[1]):
        n2 = (2*a[i]+1) * (2*a[i]+1)
        udf_data[i] = int((2*a[i]+1 - math.sqrt(n2-8*(a[i]-b[i])))/2)
