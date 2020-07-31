/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: createh5.cpp
 *
 * Creates HDF5 files that can be used for testing purposes.
 */
#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>
#include <string>

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stdout, "Syntax: %s <file.h5> [number_of_datasets]\n", argv[0]);
        return 1;
    }

    std::string hdf5_file = argv[1];
    int dataset_count = argc == 3 ? atoi(argv[2]) : 0;
    hid_t file_id = H5Fcreate(hdf5_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0)
    {
        fprintf(stderr, "Failed to create file %s\n", hdf5_file.c_str());
        return 1;
    }

    for (int count=1; count<=dataset_count; ++count)
    {
        const int dim0 = 100, dim1 = 50;
        int data[dim0][dim1];
        for (int i=0; i<dim0; ++i)
            for (int j=0; j<dim1; ++j)
                data[i][j] = count * 10 * i + j;

        char name[64];
        snprintf(name, sizeof(name)-1, "Dataset%d", count);
        hsize_t dims[2] = {dim0,dim1};
        hid_t space_id = H5Screate_simple(2, dims, NULL);
        if (space_id < 0)
        {
            fprintf(stderr, "Failed to create dataspace\n");
            return 1;
        }
        hid_t dset_id = H5Dcreate(file_id, name, H5T_STD_I32LE, space_id,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dset_id < 0)
        {
            fprintf(stderr, "Failed to create dataset\n");
            return 1;
        }
        herr_t ret = H5Dwrite(dset_id, H5T_NATIVE_INT,
            H5S_ALL, H5S_ALL, H5P_DEFAULT, data[0]);
        if (ret < 0)
        {
            fprintf(stderr, "Error writing data to file\n");
            return 1;
        }
        H5Dclose(dset_id);
        H5Sclose(space_id);
    }

    H5Fclose(file_id);
    return 0;
}
