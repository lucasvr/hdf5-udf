/*
 * Compound: expose a member of a compound as a dynamic dataset
 *
 * To embed it in an existing HDF5 file, run:
 * $ make files
 * $ hdf5-udf example-compound.h5 example-compound.cpp Temperature:1000:double
 *
 * Underneath, hdf5-udf converts the compound into a named
 * structure that can be used by getData<>(). Member names
 * are made lowercase; spaces and dashes are converted into
 * the underscore ("_") character. Last, if present, the
 * member names are truncated at the "(" or "[" characters.
 * 
 * For instance, the following compound:
 * GROUP "/" {
 *  DATASET "Dataset1" {
 *     DATATYPE  H5T_COMPOUND {
 *        H5T_STD_I64LE "Serial number";
 *        H5T_IEEE_F64LE "Temperature (F)";
 *        H5T_IEEE_F64LE "Pressure (inHg)";
 *     }
 *     DATASPACE  SIMPLE { ( 4 ) / ( 4 ) }
 *  }
 * }
 * 
 * may be converted into this named structure:
 * struct compound_dataset1 {
 *     int64_t serial_number;
 *     char _pad0[16];
 *     double temperature;
 *     double pressure;
 * };
 *
 * Note that the conversion process takes care of padding
 * the structure in case the dataset memory layout differs
 * from the storage layout.
 * 
 * The application can then simply iterate over the elements
 * retrieved from getData() and access each struct member as
 * this example shows.
 */

extern "C" void dynamic_dataset()
{
    auto compound = lib.getData<compound_dataset1>("Dataset1");
    auto udf_data = lib.getData<double>("Temperature");
    auto udf_dims = lib.getDims("Temperature");

    for (size_t i=0; i<udf_dims[0]; ++i)
    {
        udf_data[i] = compound[i].temperature;
    }
}
