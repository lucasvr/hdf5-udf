# hdf5-udf sine_wave.h5 sine_wave.py SineWave.py:100x10:int32
import math

def dynamic_dataset():
    udf_data = lib.getData("SineWave.py")
    udf_type = lib.getType("SineWave.py")
    udf_dims = lib.getDims("SineWave.py")

    N = udf_dims[0]
    M = udf_dims[1]

    for i in range(N):
        for j in range(M):
            udf_data[i*M + j] = int(math.sin(i*M + j) * 100.0)