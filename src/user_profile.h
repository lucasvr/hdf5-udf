/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: user_profile.h
 *
 * Routines to check and assure who produced a UDF.
 */
#ifndef __user_profile_h
#define __user_profile_h

#include <sys/types.h>
#include <dirent.h>
#include <glob.h>
#include <pwd.h>
#include <iostream>
#include <fstream>
#include <string>
#include "json.hpp"

using json = nlohmann::json;

// Structure Blob holds a buffer either encrypted with the user's own
// private key or decrypted with one of the public keys imported into
// the system -- it depends on the functions that use this structure.
// Metadata associated with that key are also maintained here.
struct Blob {
    Blob(uint8_t *_data, unsigned long long _size, std::string _public_key_base64="") :
        data(_data),
        size(_size),
        public_key_base64(_public_key_base64) { }

    ~Blob() {
        delete[] data;
    }

    uint8_t *data;
    unsigned long long size;
    std::string public_key_base64;
    std::string public_key_path;
    json metadata;
};

class SignatureHandler {
public:
    SignatureHandler()
    {
        const char *xdg_config_home = getenv("XDG_CONFIG_HOME");
        const char *home = getenv("HOME");
        if (xdg_config_home)
            configdir = std::string(xdg_config_home) + "/hdf5-udf/";
        else if (home)
            configdir = std::string(home) + "/.config/hdf5-udf/";
        else
        {
            auto pwp = getpwuid(getuid());
            if (pwp)
                configdir = "/tmp/hdf5-udf." + std::string(pwp->pw_name) + "/";
            else
                configdir = "/tmp/hdf5-udf";
        }
    }

    ~SignatureHandler() {}

    // Given a payload, attempt to extract it using public keys already
    // imported into the system. On success, returns a Blob object with
    // its 'data', 'size', and 'public_key_path' members set.
    Blob *extractPayload(
        const uint8_t *in,
        unsigned long long size_in,
        json &signature,
        bool first_call=true);

    // Given a payload (such as a UDF), attempt to sign it using the
    // user's own private key. Directory structures are created as
    // needed. If absent, the private key and its corresponding public
    // key are also created under @configdir.
    Blob *signPayload(const uint8_t *in, unsigned long long size_in);

    // Given the path to a public key stored in the config directory,
    // retrieve the corresponding seccomp profile rules as a JSON object.
    bool getProfileRules(std::string public_key_path, json &rules);

private:
    Blob *findPublicKey(
        const uint8_t *in,
        unsigned long long size_in,
        glob_t &globbuf);

    // Import a given public key and its metadata and save both to disk
    bool importPublicKey(json &signature);

    // Create directory structure at @configdir
    bool createDirectoryTree();

    // Save public key to disk
    bool savePublicKey(uint8_t *public_key, std::string path, bool overwrite=false);

    // Save metadata to disk
    bool saveMetadata(json j, std::string path, bool overwrite=false);

    // Save secret key to disk
    bool savePrivateKey(
        uint8_t *secret_key,
        std::string path,
        std::string meta_path,
        json &metadata);

    // Validate JSON file previously serialized by getProfileRules().
    bool validateProfileRules(std::string rulefile, json &rules);

    // Default configuration path
    std::string configdir;
};
#endif