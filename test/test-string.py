# hdf5-udf <file.h5> test-string.py Temperature.py:1000:double
def dynamic_dataset():
    input_data = lib.getData("Dataset1")
    input_dims = lib.getDims("Dataset1")
    udf_data = lib.getData("Temperature.py")
    udf_dims = lib.getDims("Temperature.py")

    for i in range(input_dims[0]):
        print(lib.string(input_data[i]), flush=True)

    for i in range(udf_dims[0]):
        udf_data[i] = i * 1.0
