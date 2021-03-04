// hdf5-udf sine_wave.h5 sine_wave.cpp SineWave.cpp:100x100:int32
extern "C" void dynamic_dataset()
{
    auto udf_data = lib.getData<int32_t>("SineWave.cpp");
    auto udf_type = lib.getType("SineWave.cpp");
    auto udf_dims = lib.getDims("SineWave.cpp");

    auto N = udf_dims[0];
    auto M = udf_dims[1];

    for (size_t i=0; i<N; ++i)
        for (size_t j=0; j<M; ++j)
            udf_data[i*M + j] = int(sinf(i*M + j) * 100.0);
}
