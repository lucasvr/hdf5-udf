#
# Shows how to generate a compound dataset from a UDF
#
# To embed this UDF in an existing HDF5 file, run:
# $ make files
# $ hdf5-udf example-compound.h5 example-output_compound.py 'Observations:{id:uint32,location:string,temperature:float}:1000'
#

def dynamic_dataset():
    udf_data = lib.getData("Observations")
    udf_dims = lib.getDims("Observations")

    for i in range(udf_dims[0]):
        udf_data[i].id = i

        # Here we can either write directly to udf_data[i].location
        # or use the lib.setString() API. The latter is preferred
        # as it prevents writes outside the boundaries of the buffer.
        #udf_data[i].location = "location_{}".format(i).encode("utf-8")
        lib.setString(udf_data[i].location, "location_{}".format(i).encode("utf-8"))

        udf_data[i].temperature = 100.0 + i