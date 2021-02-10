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
#include <wchar.h>
#include <hdf5.h>
#include <string>
#include <cstring>

typedef struct { uint8_t r; uint8_t g; uint8_t b; } pal_t;

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stdout, "Syntax: %s <file.h5> <dataset> [palette dataset]\n", argv[0]);
        return 1;
    }

    std::string hdf5_file = argv[1];
    std::string hdf5_dataset = argv[2];
    std::string hdf5_palette = argc > 3 ? argv[3] : "";

    // Enable printing UTF-8 characters
    setlocale(LC_ALL, "en_US.UTF-8");

    hid_t file_id = H5Fopen(hdf5_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0)
    {
        fprintf(stderr, "Failed to open file %s\n", hdf5_file.c_str());
        return 1;
    }

    // Read dataset
    hid_t dataset_id = H5Dopen(file_id, hdf5_dataset.c_str(), H5P_DEFAULT);
    if (dataset_id < 0)
    {
        fprintf(stderr, "Failed to open dataset %s\n", hdf5_dataset.c_str());
        return 1;
    }

    hid_t type_id = H5Dget_type(dataset_id);
    hid_t space_id = H5Dget_space(dataset_id);
    size_t datatype_size = H5Tget_size(type_id);
    int ndims = H5Sget_simple_extent_ndims(space_id);
    if (ndims != 2)
    {
        fprintf(stderr, "Expected a dataset with 2 dimensions, got %d instead\n", ndims);
        return 1;
    }
    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(space_id, dims, NULL);

    uint8_t *rdata = new uint8_t[dims[0] * dims[1] * datatype_size];
    if (H5Tequal(type_id, H5T_NATIVE_INT))
        H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);
    else if (H5Tequal(type_id, H5T_NATIVE_UINT8))
        H5Dread(dataset_id, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);
    else
    {
        fprintf(stderr, "We're only ready to deal with INT/UINT8 data types, sorry!\n");
        return 1;
    }
    H5Sclose(space_id);
    H5Tclose(type_id);
    H5Dclose(dataset_id);

    // Read palette, if provided. We assume it's an uint8_t
    uint8_t *pal = NULL;
    hsize_t pal_dims = 0;
    if (hdf5_palette.size())
    {
        hid_t palette_id = H5Dopen(file_id, hdf5_palette.c_str(), H5P_DEFAULT);
        hid_t pal_space_id = H5Dget_space(palette_id);
        int ndims = H5Sget_simple_extent_ndims(pal_space_id);
        if (ndims != 1)
        {
            fprintf(stderr, "Expected a palette with 1 dimension, got %d instead\n", ndims);
            return 1;
        }
        H5Sget_simple_extent_dims(pal_space_id, &pal_dims, NULL);

        pal = new uint8_t[pal_dims];
        memset(pal, 0, pal_dims);
        H5Dread(palette_id, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, pal);
        H5Sclose(pal_space_id);
        H5Dclose(palette_id);
    }
    H5Fclose(file_id);

    // Image-based datasets needs to be stretched a little bit so they look prettier on
    // the console. They also look better when rendered as special characters instead
    // of a collection of 0s and 1s.
    bool is_tux = hdf5_dataset.compare("Tux") == 0;

    char line[8192];
    char *cur = line, *end = line + sizeof(line);
    memset(line, 0, sizeof(line));
    for (size_t i=0; i<dims[0]*dims[1]; ++i)
    {
        if (i && i%dims[0] == 0)
        {
            printf("%s\n", line);
            memset(line, 0, sizeof(line));
            cur = line;
        }

        if (is_tux)
        {
            wchar_t shade = 0x2592;
            int *rdata_int = (int *) rdata;
            cur += snprintf(cur, end-cur, "%lc%lc", rdata_int[i] == 1 ? ' ' : shade, rdata_int[i] == 1 ? ' ' : shade);
        }
        else if (hdf5_palette.size())
        {
            pal_t *pal_ptr = (pal_t *) pal;
            uint8_t data = rdata[i];
            if (pal_ptr)
            {
                // Render this pixel using Truecolor ANSI escape codes
                pal_t *color = &pal_ptr[data];
                cur += snprintf(cur, end-cur, "\033[48;2;%d;%d;%dm**\033[0m", color->r, color->g, color->b);
            }
            else
                cur += snprintf(cur, end-cur, "%c%c", data, data);
        }
        else
            cur += snprintf(cur, end-cur, "%d", (int) rdata[i]);
    }
    printf("%s\n", line);

    return 0;
}
