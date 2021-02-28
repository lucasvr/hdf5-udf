/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: user_profile.h
 *
 * Routines to check and assure who produced a UDF.
 */
#ifndef __user_profile_h
#define __user_profile_h

#include <dirent.h>
#include <iostream>
#include <fstream>
#include <string>

// Structure Blob holds a buffer either encrypted with the user's own
// private key or decrypted with one of the public keys imported into
// the system -- it depends on the functions that use this structure.
// Metadata associated with that key are also maintained here.
struct Blob {
    Blob(uint8_t *_data, unsigned long long _size, std::string _path) :
        data(_data),
        size(_size),
        path(_path)
    {
    }

    ~Blob()
    {
        delete[] data;
    }

    uint8_t *data;
    unsigned long long size;
    std::string path;
};

class SignatureHandler {
public:
    SignatureHandler()
    {
        const char *home = getenv("HOME");
        if (home)
            configdir = std::string(home) + "/.config/hdf5-udf/";
        else
            configdir = "/tmp/hdf5-udf/";
    }

    ~SignatureHandler() {}

    // Given a payload, attempt to extract it using public keys already
    // imported into the system.
    Blob *extractPayload(const uint8_t *in, unsigned long long size_in);

    // Given a payload (such as a UDF), attempt to sign it using the
    // user's own private key. Directory structures are created as
    // needed. If absent, the private key and its corresponding public
    // key are also created under @configdir.
    Blob *signPayload(const uint8_t *in, unsigned long long size_in);

private:
    // Create directory structure at @configdir
    bool createDirectoryTree();

    // Save public key to disk
    bool savePublicKey(uint8_t *public_key, std::string path);

    // Save secret key to disk
    bool savePrivateKey(uint8_t *secret_key, std::string path);

    // Default configuration path
    std::string configdir;
};
#endif