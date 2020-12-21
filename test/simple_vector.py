# hdf5-udf simple_vector.h5 simple_vector.py Simple.py:1500:float
def dynamic_dataset():
    udf_data = lib.getData("Simple.py")
    udf_dims = lib.getDims("Simple.py")
    for i in range(udf_dims[0]):
        udf_data[i] = i
