/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: user_profile.cpp
 *
 * Directory and file instructions to validate user credentials
 */
#include <hdf5.h>
#include <stdlib.h>
#include <string.h>
#include <sodium.h>
#include <glob.h>
#include <sys/stat.h>

#include "user_profile.h"

using namespace std;

bool UserSignature::initialize()
{
    if (sodium_init() == -1)
    {
        fprintf(stderr, "Libsodium could not be initialized\n");
        return false;
    }
    return true;
}

bool UserSignature::validateKey()
{
    if (!initialize())
    {
        fprintf(stderr, "Libsodium error\n");
        return false;
    }
    glob_t globbuf;
    string public_keys = DIR_PATH + "public.key";
    int ret = glob(public_keys.c_str(), 0, NULL, &globbuf);
    if (ret != 0)
    {
        fprintf(stderr, "glob error: %d (%s)\n", ret, strerror(ret));
        return false;
    }

    for (size_t i=0; i<globbuf.gl_pathc; ++i)
    {
        string full_path = globbuf.gl_pathv[i];

        // Read public key
        unsigned char* pub_key;
        streampos file_size;
        ifstream file(full_path, ios::in|ios::binary|ios::ate);
        if (file.is_open())
        {
            file_size = file.tellg();
            pub_key = new unsigned char[file_size];
            file.seekg(0, ios::beg);
            file.read((char*) pub_key, file_size);
        }
        else
        {
            continue;
        }

        full_path = DIR_PATH + "/udf.signed";

        // Read signed file
        ifstream signed_file(full_path, ios::in|ios::binary|ios::ate);
        unsigned char* signed_message;
        streampos signed_size;

        if (signed_file.is_open())
        {
            bool stop = false;
            signed_size = signed_file.tellg();
            signed_message = new unsigned char[signed_size];
            signed_file.seekg(0, ios::beg);
            signed_file.read((char*) signed_message, signed_size);
            unsigned char* unsigned_message;
            unsigned_message = new unsigned char[static_cast<size_t>(signed_size)];
            unsigned long long unsigned_message_len;

            if (crypto_sign_open(unsigned_message, &unsigned_message_len,
                signed_message, (int) signed_size, pub_key) != 0)
            {
                printf("Keys do not match, will try with another public key\n");
            }
            else
                stop = true;
            delete[] pub_key;
            delete[] signed_message;
            delete[] unsigned_message;
            if (stop)
                break;
        }
        else
        {
            fprintf(stderr, "Error opening %s\n", globbuf.gl_pathv[i]);
            delete[] pub_key;
            return false;
        }
    }
    return true;
}

bool UserSignature::signFile(string udf_file)
{
    if (!initialize())
    {
        fprintf(stderr, "Libsodium error\n");
        return false;
    }

    unsigned char pk[crypto_sign_PUBLICKEYBYTES];
    unsigned char sk[crypto_sign_SECRETKEYBYTES];
    unsigned char* signed_message;
    unsigned long long signed_message_len;
    string stored_key = DIR_PATH + "private.key";
    struct stat buf;
    if (stat(stored_key.c_str(), &buf) == 0)
    {
        ifstream key_file(stored_key.c_str(), ios::in|ios::binary|ios::ate);
        if (key_file.is_open())
        {
            key_file.seekg(0, ios::beg);
            key_file.read((char*)sk, crypto_sign_SECRETKEYBYTES);
        }
        else
        {
            fprintf(stderr, "Could not open %s. Error:\n %s\n", stored_key.c_str(), strerror(errno));
            return false;
        }
        if (crypto_sign_ed25519_sk_to_pk(pk, sk) != 0)
        {
            fprintf(stderr, "Error extracting public key\n");
            return false;
        }
        
        char *udf = (char *) calloc(buf.st_size, sizeof(char));
        FILE *fp = fopen(udf_file.c_str(), "r");
        if (udfFromFile(udf, fp, udf_file.c_str(), buf))
        {
            signed_message = new unsigned char[crypto_sign_BYTES + buf.st_size];
            if (crypto_sign(signed_message, &signed_message_len,
            reinterpret_cast<const unsigned char *>(udf), buf.st_size, sk) != 0)
            {
                fprintf(stderr, "Could not sign message\n");
                delete[] signed_message;
                return false;
            }
            return createSignature(signed_message, signed_message_len);
        }
        return false;
    }
    else
    {
        if (crypto_sign_keypair(pk, sk))
        {
            fprintf(stderr, "Error generating public and secret key\n");
            return false;
        }
    }

    struct stat statbuf;
    if (stat(udf_file.c_str(), &statbuf) < 0)
    {
        perror(udf_file.c_str());
        return false;
    }
    char *udf = (char *) calloc(statbuf.st_size, sizeof(char));
    FILE *fp = fopen(udf_file.c_str(), "r");
    if (udfFromFile(udf, fp, udf_file.c_str(), statbuf))
    {
        signed_message = new unsigned char[crypto_sign_BYTES + statbuf.st_size];
        if (crypto_sign(signed_message, &signed_message_len,
        reinterpret_cast<const unsigned char *>(udf), statbuf.st_size, sk) != 0)
        {
            fprintf(stderr, "Could not sign message\n");
            delete[] signed_message;
            return false;
        }
        return createDirectoryTree(pk, sk, signed_message, signed_message_len);
    }
    return false;
}

bool UserSignature::createDirectoryTree(unsigned char* pk, unsigned char* sk, 
    unsigned char* signed_message, unsigned long long signed_message_len)
{
    DIR *pDir;
    pDir = opendir((DIR_PATH).c_str());
    bool returnvalue = true;
    if (pDir == NULL)
    {
        // Creating a directory
        if (mkdir((DIR_PATH).c_str(), 0777) == -1)
        {
            cerr << "Error: " << strerror(errno) << endl;
            returnvalue = false;
        }
        if (mkdir((DIR_PATH+"/default").c_str(), 0755) == -1)
        {
            cerr << "Error: " << strerror(errno) << endl;
            returnvalue = false;
        }
        if (mkdir((DIR_PATH+"/allow").c_str(), 0755) == -1)
        {
            cerr << "Error: " << strerror(errno) << endl;
            returnvalue = false;
        }
        if (mkdir((DIR_PATH+"/deny").c_str(), 0755) == -1)
        {
            cerr << "Error: " << strerror(errno) << endl;
            returnvalue = false;
        }
    }
    closedir(pDir);
    if (returnvalue)
    {
        if (createPublicKey(pk) && createSecretKey(sk) && createSignature(signed_message, signed_message_len))
            returnvalue = true;
        else
            returnvalue = false;
    }
    delete[] signed_message;
    return returnvalue;
}

bool UserSignature::createPublicKey(unsigned char* pk)
{
    string full_path = DIR_PATH + "public" + ".key";
    ifstream file(full_path);
    if (!file.is_open())
    {
        ofstream file(full_path);
        file.write((char *) pk, crypto_sign_PUBLICKEYBYTES);
    }
    else
    {
        fprintf(stderr, "Public key %s already exists\n", full_path.c_str());
        return false;
    }
    return true;
}

bool UserSignature::createSecretKey(unsigned char* sk)
{
    string full_path = DIR_PATH + "private" + ".key";
    ifstream file(full_path);
    if (!file.is_open())
    {
        ofstream file(full_path);
        file.write((char *) sk, crypto_sign_SECRETKEYBYTES);
    }
    else
    {
        fprintf(stderr, "Secret key %s already exists\n", full_path.c_str());
        return false;
    }
    return true;
}

bool UserSignature::createSignature(unsigned char* signed_message, unsigned long long signed_message_len)
{
    string full_path = DIR_PATH + "/udf.signed";
    ifstream infile(full_path);
    if (!infile.is_open())
    {
        ofstream outfile(full_path);
        outfile.write((char *) signed_message, signed_message_len);
    }
    else
    {
        ofstream outfile(full_path, ofstream::trunc);
        outfile.write((char *) signed_message, signed_message_len);
    }
    return true;
}

bool UserSignature::udfFromFile(char* udf, FILE *fp, const char* udf_file, struct stat statbuf)
{
    if (!fp)
    {
        fprintf(stderr, "Could not open %s. Error: %s\n", udf_file, strerror(errno));
        return false;
    }
    fread(udf, sizeof(char), statbuf.st_size, fp);
    fclose(fp);
    return true;
}