/*
 * Shows how to generate a compound dataset from a UDF
 *
 * To embed this UDF in an existing HDF5 file, run:
 * $ make files
 * $ hdf5-udf compound.h5 test-compound-output.cpp 'Observations.cpp:{id:uint32,location:string,temperature:float}:1000'
 */

extern "C" void dynamic_dataset()
{
    auto udf_data = lib.getData<observations_cpp_t>("Observations.cpp");
    auto udf_dims = lib.getDims("Observations.cpp");

    for (size_t i=0; i<udf_dims[0]; ++i)
    {
        udf_data[i].id = i;
        snprintf(udf_data[i].location, sizeof(udf_data[i].location), "location_%d", i);
        udf_data[i].temperature = 100.0 + i;
    }
}