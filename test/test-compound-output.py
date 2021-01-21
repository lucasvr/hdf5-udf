#
# Shows how to generate a compound dataset from a UDF
#
# To embed this UDF in an existing HDF5 file, run:
# $ make files
# $ hdf5-udf compound.h5 test-compound-output.py 'Observations.py:{id:uint32,location:string,temperature:float}:1000'
#

def dynamic_dataset():
    udf_data = lib.getData("Observations.py")
    udf_dims = lib.getDims("Observations.py")

    for i in range(udf_dims[0]):
        udf_data[i].id = i
        udf_data[i].location = "location_{}".format(i).encode("utf-8")
        udf_data[i].temperature = 100.0 + i