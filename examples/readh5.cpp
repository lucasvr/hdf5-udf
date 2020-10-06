/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: readh5.cpp
 *
 * Dumps a HDF5 dataset to stdout. This utility has been written as a visualization
 * helper for the dataset produced by example-socket.cpp. It may work with other
 * datasets too, as long as they're small enough and that they're encoded as int32.
 */
#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>
#include <string>
#include <cstring>

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stdout, "Syntax: %s <file.h5> <dataset>\n", argv[0]);
        return 1;
    }

    std::string hdf5_file = argv[1];
	std::string hdf5_dataset = argv[2];
    hid_t file_id = H5Fopen(hdf5_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0)
    {
        fprintf(stderr, "Failed to open file %s\n", hdf5_file.c_str());
        return 1;
    }
	hid_t dataset_id = H5Dopen(file_id, hdf5_dataset.c_str(), H5P_DEFAULT);
	if (dataset_id < 0)
	{
        fprintf(stderr, "Failed to open dataset %s\n", hdf5_dataset.c_str());
        return 1;
    }

	hid_t type_id = H5Dget_type(dataset_id);
	if (! H5Tequal(type_id, H5T_NATIVE_INT))
	{
		fprintf(stderr, "We're only ready to deal with native INT data types, sorry!\n");
		return 1;
	}

	hsize_t dims[2] = {0, 0};
	hid_t space_id = H5Dget_space(dataset_id);
	int ndims = H5Sget_simple_extent_ndims(space_id);
	if (ndims != 2)
	{
		fprintf(stderr, "Expected a dataset with 2 dimensions, got %d instead\n", ndims);
		return 1;
	}
	H5Sget_simple_extent_dims(space_id, dims, NULL);

	uint32_t *rdata = new uint32_t[dims[0] * dims[1]];
	memset(rdata, 0, sizeof(uint32_t) * dims[0] * dims[1]);
	H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);

	// The Tux dataset needs to be stretched a little bit so it looks prettier
	// on the console. It also looks better when rendered as solid blocks instead
	// of a collection of 0s and 1s.
	bool is_tux = hdf5_dataset.compare("Tux") == 0;
	for (size_t i=0; i<dims[0]*dims[1]; ++i) {
		if (i && i%dims[0] == 0)
			printf("\n");
		if (is_tux)
			printf("%s", rdata[i] == 1 ? "  " : "\xE2\x96\x92\xE2\x96\x92");
		else
			printf("%d", rdata[i]);
	}
	printf("\n");

    return 0;
}
