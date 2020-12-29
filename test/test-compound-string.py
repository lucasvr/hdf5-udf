# hdf5-udf <file.h5> test-compound-string.py Temperature.py:1000:double
def dynamic_dataset():
    compound = lib.getData("Dataset1")
    udf_data = lib.getData("Temperature.py")
    udf_dims = lib.getDims("Temperature.py")

    for i in range(udf_dims[0]):
        print("serial: {}, location: {}, temperature: {:.6f}, pressure: {:.6f}".format(
            compound[i].serial_number,
            lib.string(compound[i].location),
            compound[i].temperature,
            compound[i].pressure), flush=True)
        udf_data[i] = compound[i].temperature
