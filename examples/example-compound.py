#
# Compound: expose a member of a compound as a dynamic dataset
#
# To embed it in an existing HDF5 file, run:
# $ make files
# $ hdf5-udf example-compound.h5 example-compound.py Temperature:1000:double
#
# Underneath, hdf5-udf converts the compound into a named
# class that can be used by getData<>() through the Foreign
# Function Interface. Member names are made lowercase; spaces
# and dashes are converted into the underscore ("_") character.
# Last, if present, the member names are truncated at the "("
# or "[" characters.
# 
# For instance, the following compound:
# GROUP "/" {
#  DATASET "DS1" {
#     DATATYPE  H5T_COMPOUND {
#        H5T_STD_I64LE "Serial number";
#        H5T_IEEE_F64LE "Temperature (F)";
#        H5T_IEEE_F64LE "Pressure (inHg)";
#     }
#     DATASPACE  SIMPLE { ( 4 ) / ( 4 ) }
#  }
# }
# 
# is converted into the following structure:
# struct dataset1_t {
#     int64_t serial_number;
#     double temperature;
#     double pressure;
# };
#
# The application can then simply iterate over the elements
# retrieved from getData() and access each struct member as
# this example shows.
#

def dynamic_dataset():
    compound = lib.getData("Dataset1")
    udf_data = lib.getData("Temperature")
    udf_dims = lib.getDims("Temperature")

    for i in range(udf_dims[0]):
        udf_data[i] = compound[i].temperature
