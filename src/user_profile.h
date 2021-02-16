/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: user_profile.h
 *
 * Interfaces for user/UDF privileges and key checks.
 */
#ifndef __user_profile_h
#define __user_profile_h

class KeyChecks {
    const std::string DIR_PATH = std::string(getenv("HOME")) + "/.config/hdf5-udf/";

    public:
    // Check if libsodium is properly set
    int initialize();

    // Validate key files
    int validate_key();
    
    // Sign files and create directories structure
    int sign_file(std::string udf_file);
};

class Directories{
    const std::string DIR_PATH = std::string(getenv("HOME")) + "/.config/hdf5-udf/";

    public:
    void create_dir_struct(unsigned char* pk, unsigned char* sk, 
    unsigned char* signed_message, unsigned long long signed_message_len);
};

#endif