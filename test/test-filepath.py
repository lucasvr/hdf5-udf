# hdf5-udf <file.h5> test-filepath.py Temperature.py:1000:double
def dynamic_dataset():
    path = lib.getFilePath().decode('utf-8').split('/')[-1]
    print(f"HDF5 file path is '{path}'", flush=True)
