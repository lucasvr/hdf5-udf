/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: user_profile.h
 *
 * Routines to check and assure who is writing the UDF
 * and give correct access rights to the user
 */
#ifndef __user_profile_h
#define __user_profile_h

#include <dirent.h>
#include <iostream>
#include <fstream>

class UserSignature {
    std::string DIR_PATH;       // Default configuration files directory path

    public:
    // Override constructor to assign value to DIR_PATH
    UserSignature()
    {
        if (getenv("HOME")) 
            DIR_PATH = std::string(getenv("HOME")) + "/.config/hdf5-udf/";
        else
            DIR_PATH = "/tmp/hdf5-udf/";
    }

    ~UserSignature() {}

    // Check if libsodium is properly set
    bool initialize();

    // Validate key files
    bool validateKey();
    
    // Sign files and create directories structure
    bool signFile(std::string udf_file);

    // Create desired directory structure and key files if not exists 
    bool createDirectoryTree(unsigned char* pk, unsigned char* sk, 
    unsigned char* signed_message, unsigned long long signed_message_len);

    private:
    // Create public key file if not exists 
    bool createPublicKey(unsigned char* pk);

    // Create secret key file if not exists 
    bool createSecretKey(unsigned char* sk);

    // Create signed udf file if not exists or overwrite the existing one
    bool createSignature(unsigned char* signed_message, unsigned long long signed_message_len);

    // Read UDF file to char*
    bool udfFromFile(char* udf, FILE *fp, const char* udf_file, struct stat statbuf);

};
#endif