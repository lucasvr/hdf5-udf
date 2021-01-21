#
# Shows how to generate a compound dataset from a UDF given
# a CSV file as input. Note that we use file I/O, which is
# disallowed by default. You will have to build the software
# without support for sandboxing (i.e., `make OPT_SANDBOX=0`)
# in order to run this example.
#
# The input file used in this example is `albumlist.csv`,
# retrieved from https://github.com/Currie32/500-Greatest-Albums
#
# To embed this UDF in an existing HDF5 file, run:
# $ make files
# $ hdf5-udf example-compound.h5 example-csv_to_compound.py \
#   'GreatestAlbums:{id:int32,year:int16,album:string,artist:string,genre:string,subgenre:string}:500'
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

            # Note: because the strings are limited to 32 characters we
            # have to make sure we don't copy more data than allowed.
            # If we ignore that limitation, FFI will throw an exception
            # and our UDF will fail to run.
            udf_data[i].id = int(parts[0])
            udf_data[i].year = int(parts[1])
            udf_data[i].album = parts[2].encode("utf-8")[:32]
            udf_data[i].artist = parts[3].encode("utf-8")[:32]
            udf_data[i].genre = parts[4].encode("utf-8")[:32]
            udf_data[i].subgenre = parts[5].encode("utf-8")[:32]