#
# Shows how to generate a compound dataset from a UDF given
# a CSV file as input.
#
# The input file used in this example is `albumlist.csv`,
# retrieved from https://github.com/Currie32/500-Greatest-Albums
#
# To embed this UDF in an existing HDF5 file, run:
# $ make files
# $ hdf5-udf example-compound.h5 example-csv_to_compound.py \
#   'GreatestAlbums:{id:int32,year:int16,album:string,artist:string,genre:string,subgenre:string}:500'
#
# Note that we're omitting the size of the string members. That
# instructs HDF5-UDF to use the default size of 32 characters for
# each of them. Please adjust these values according to the input
# data you provide to the UDF.
#
# ---------
# IMPORTANT
# ---------
#
# This UDF uses file I/O, which is disallowed by the sandbox.
# For the time being, in order to run this test you will have
# to build the software without support for sandboxing (i.e.,
# running `make OPT_SANDBOX=0`).
#


def dynamic_dataset():
    udf_data = lib.getData("GreatestAlbums")
    udf_dims = lib.getDims("GreatestAlbums")

    # The file is encoded as ISO-8859-1, so instruct Python about it
    with open("albumlist.csv", encoding="iso-8859-1") as f:

        # Read and ignore the header
        f.readline()

        for i, line in enumerate(f.readlines()):
            # Remove double-quotes and newlines around certain strings
            parts = [col.strip('"').strip("\n") for col in line.split(",")]

            # Note: unless we specify 'string(N)' with a large enough N, we
            # may end up attempting to write more characters into the string
            # buffer than allowed. Here we use the lib.setString() API so it
            # performs boundary checks for us. The alternative is to write
            # directly to each udf_data[i] member, at the risk of receiving
            # a FFI exception if more data is attempted to be copied than
            # allowed. Again, note that, when '(N)' is absent, the default
            # string member size is of 32 characters.
            udf_data[i].id = int(parts[0])
            udf_data[i].year = int(parts[1])
            lib.setString(udf_data[i].album,  parts[2].encode("utf-8"))
            lib.setString(udf_data[i].artist,  parts[3].encode("utf-8"))
            lib.setString(udf_data[i].genre,  parts[4].encode("utf-8"))
            lib.setString(udf_data[i].subgenre,  parts[5].encode("utf-8"))