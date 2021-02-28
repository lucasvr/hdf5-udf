/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: user_profile.cpp
 *
 * Management of user credentials
 */
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sodium.h>
#include <hdf5.h>
#include <glob.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <pwd.h>
#include <string>
#include <vector>
#include "user_profile.h"

using namespace std;

int globError(const char *epath, int eerrno)
{
    if (eerrno != ENOENT)
    {
        // Report error and stop trying to glob remaining files
        fprintf(stderr, "Error reading file %s: %s\n", epath, strerror(eerrno));
        return -1;
    }
    return 0;
}

const char *globStrError(int errnum)
{
    if (errnum == GLOB_NOSPACE)
        return "out of memory";
    else if (errnum == GLOB_ABORTED)
        return "read error";
    else if (errnum == GLOB_NOMATCH)
        return "no matches found";
    else
        return "unknown error";
}

Blob *SignatureHandler::extractPayload(const uint8_t *in, unsigned long long size_in)
{
    if (sodium_init() == -1)
    {
        fprintf(stderr, "Failed to initialize libsodium\n");
        return NULL;
    }

    // User's own public key is stored as $configdir/*.pub
    // Public keys from other users imported into the system
    // are stored as $configdir/{default,allow,deny,...}/*.pub
    glob_t globbuf;
    string public_keys = configdir + "{*.pub,*/*.pub}";
    int ret = glob(public_keys.c_str(), GLOB_BRACE, globError, &globbuf);
    if (ret != 0)
    {
        fprintf(stderr, "Error scanning %s: %s\n", public_keys.c_str(), globStrError(ret));
        return NULL;
    }

    for (size_t i=0; i<globbuf.gl_pathc; ++i)
    {
        // Read public key
        ifstream file(globbuf.gl_pathv[i], ios::in|ios::binary|ios::ate);
        if (! file.is_open())
        {
            fprintf(stderr, "Error opening %s: %s\n", globbuf.gl_pathv[i], strerror(errno));
            continue;
        }
        auto file_size = file.tellg();
        file.seekg(0, ios::beg);
        auto pub_key = new uint8_t[file_size];
        file.read((char *) pub_key, file_size);

        // Attempt to decrypt the input data with the public key read
        auto data = new uint8_t[size_in];
        auto size = size_in;
        auto path = globbuf.gl_pathv[i];
        if (crypto_sign_open(data, &size, in, size_in, pub_key) != 0)
        {
            const char *extra = i == globbuf.gl_pathc-1 ? "" : ", will try another key";
            fprintf(stderr, "Could not decode UDF with %s%s\n", path, extra);
            delete[] pub_key;
            delete[] data;
            continue;
        }
        delete[] pub_key;
        globfree(&globbuf);
        return new Blob(data, size, path);
    }

    globfree(&globbuf);
    return NULL;
}

Blob *SignatureHandler::signPayload(const uint8_t *in, unsigned long long size_in)
{
    if (sodium_init() == -1)
    {
        fprintf(stderr, "Failed to initialize libsodium\n");
        return NULL;
    }

    uint8_t public_key[crypto_sign_PUBLICKEYBYTES];
    uint8_t secret_key[crypto_sign_SECRETKEYBYTES];
    std::string path;

    // User's own private key is stored as $configdir/*.priv
    // Note that we pick the first one returned by glob().
    glob_t globbuf;
    string private_keys = configdir + "*.priv";
    int ret = glob(private_keys.c_str(), GLOB_BRACE, globError, &globbuf);
    if (ret == 0)
    {
        path = globbuf.gl_pathv[0];
        globfree(&globbuf);

        // Read private key from the file
        ifstream file(path.c_str(), ios::in|ios::binary|ios::ate);
        if (! file.is_open())
        {
            fprintf(stderr, "Error opening private key %s\n", path.c_str());
            return NULL;
        }
        else if (file.tellg() != crypto_sign_SECRETKEYBYTES)
        {
            fprintf(stderr, "Error: private key %s has %d bytes, expected %u instead\n",
                path.c_str(), static_cast<int>(file.tellg()), crypto_sign_SECRETKEYBYTES);
            return NULL;
        }
        file.seekg(0, ios::beg);
        file.read((char *) secret_key, crypto_sign_SECRETKEYBYTES);

        // Extract the public key from the secret key
        if (crypto_sign_ed25519_sk_to_pk(public_key, secret_key) != 0)
        {
            fprintf(stderr, "Error extracting public key from secret %s\n", path.c_str());
            return NULL;
        }
    }
    else if (ret == GLOB_NOMATCH)
    {
        // Create new public and private keys
        if (crypto_sign_keypair(public_key, secret_key))
        {
            fprintf(stderr, "Error generating public and secret keys\n");
            return NULL;
        }

        // Save both keys to disk
        auto pw = getpwuid(getuid());
        std::string private_path = configdir + (pw ? pw->pw_name : "my") + ".priv";
        std::string public_path = configdir + (pw ? pw->pw_name : "my") + ".pub";
        createDirectoryTree();
        savePrivateKey(secret_key, private_path);
        savePublicKey(public_key, public_path);

        path = private_path;
    }
    else if (ret != 0)
    {
        fprintf(stderr, "Error scanning %s: %s\n", private_keys.c_str(), globStrError(ret));
        return NULL;
    }

    // Sign the payload
    unsigned long long signed_message_len;
    auto signed_message = new uint8_t[crypto_sign_BYTES + size_in];
    if (crypto_sign(signed_message, &signed_message_len, in, size_in, secret_key) != 0)
    {
        fprintf(stderr, "Could not sign message\n");
        delete[] signed_message;
        return NULL;
    }

    return new Blob(signed_message, signed_message_len, path);
}

bool SignatureHandler::createDirectoryTree()
{
    // Create config directory tree
    std::vector<std::string> dirs;
    dirs.push_back(configdir);
    dirs.push_back(configdir + "default");
    dirs.push_back(configdir + "allow");
    dirs.push_back(configdir + "deny");

    struct stat statbuf;
    for (auto &dir: dirs)
        if (stat(dir.c_str(), &statbuf) < 0 && errno == ENOENT)
            if (mkdir(dir.c_str(), 0755) < 0)
            {
                fprintf(stderr, "Error creating directory %s: %s\n", dir.c_str(), strerror(errno));
                return false;
            }
    return true;
}

bool SignatureHandler::savePublicKey(uint8_t *public_key, std::string path)
{
    ifstream file(path);
    if (! file.is_open())
    {
        ofstream file(path);
        file.write((char *) public_key, crypto_sign_PUBLICKEYBYTES);
    }
    return true;
}

bool SignatureHandler::savePrivateKey(uint8_t *secret_key, std::string path)
{
    ifstream file(path);
    if (! file.is_open())
    {
        ofstream file(path);
        file.write((char *) secret_key, crypto_sign_SECRETKEYBYTES);
    }
    return true;
}