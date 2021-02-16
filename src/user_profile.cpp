#include <dirent.h>
#include <hdf5.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sodium.h>
#include <glob.h>
#include <sys/stat.h>

#include "user_profile.h"

using namespace std;

int KeyChecks::initialize(){
    if (sodium_init() == -1) {
        fprintf(stderr, "Libsodium could not be initialized\n");
        return -1;
    }
    return 0;
}

int KeyChecks::validate_key()
{
    if(initialize() == -1)
        return -1;
    
    glob_t globbuf;
    string public_keys = DIR_PATH + "public.key";
    printf("glob pattern: %s\n", public_keys.c_str());
    int ret = glob(public_keys.c_str(), 0, NULL, &globbuf);

    if (ret != 0) {
        fprintf(stderr, "glob error: %d (%s)\n", ret, strerror(ret));
        return -1;
    }

    for (size_t i=0; i<globbuf.gl_pathc; ++i) {
        printf("Scanning %s (%ld/%ld)\n", globbuf.gl_pathv[i], i, globbuf.gl_pathc);
        string full_path = globbuf.gl_pathv[i];

        // Read public key
        unsigned char* pub_key;
        streampos file_size;
        ifstream file(full_path, ios::in|ios::binary|ios::ate);
        if(file.is_open()){
            file_size = file.tellg();
            pub_key = new unsigned char[file_size];
            file.seekg (0, ios::beg);
            file.read((char*)pub_key, file_size);
        }

        full_path = DIR_PATH + "/udf.signed";

        // Read signed file
        ifstream signed_file(full_path, ios::in|ios::binary|ios::ate);
        unsigned char* signed_message;
        streampos signed_size;
                
        if(signed_file.is_open()){
            signed_size = signed_file.tellg();
            signed_message = new unsigned char[signed_size];

            signed_file.seekg (0, ios::beg);
            signed_file.read ((char*)signed_message, signed_size); 
            unsigned char unsigned_message[static_cast<size_t>(signed_size)];
            unsigned long long unsigned_message_len;

            if (crypto_sign_open(unsigned_message, &unsigned_message_len,
                signed_message, (int) signed_size, pub_key) != 0) {
                printf("Keys do not match, will try with another public key\n");
            }

            else {
                printf("Keys match!\n");
                break;
            }
        }
        else {
            fprintf(stderr, "Error opening udf.signed\n");
            return -1;
        }
    }
    return 0;
}

int KeyChecks::sign_file(string udf_file)
{

    if(initialize() == -1)
        return -1;

    unsigned char pk[crypto_sign_PUBLICKEYBYTES];
    unsigned char sk[crypto_sign_SECRETKEYBYTES];
    unsigned char* signed_message;

    string stored_key = DIR_PATH + "private.key";
    struct stat buf;
    if (stat(stored_key.c_str(), &buf) == 0){
        ifstream key_file(stored_key.c_str(), ios::in|ios::binary|ios::ate);

        if(key_file.is_open()){
            key_file.seekg (0, ios::beg);
            key_file.read ((char*)sk, crypto_sign_SECRETKEYBYTES); 
        }

        crypto_sign_ed25519_sk_to_pk(pk, sk);
    }
    else{    
        crypto_sign_keypair(pk, sk);
    }

    struct stat statbuf;
    if (stat(udf_file.c_str(), &statbuf) < 0) {
        perror(udf_file.c_str());
        return -1;
    }
    char *udf = (char *) calloc(statbuf.st_size, sizeof(char));
    FILE *fp = fopen(udf_file.c_str(), "r");
    fread(udf, sizeof(char), statbuf.st_size, fp);
    fclose(fp);
        
    signed_message = new unsigned char[crypto_sign_BYTES + statbuf.st_size];
    unsigned long long signed_message_len;
    crypto_sign(signed_message, &signed_message_len,
        reinterpret_cast<const unsigned char *>(udf), statbuf.st_size, sk);

    Directories d;
    d.create_dir_struct(pk, sk, signed_message, signed_message_len);
    
    return 0;
}

void Directories::create_dir_struct(unsigned char* pk, unsigned char* sk, 
    unsigned char* signed_message, unsigned long long signed_message_len)
{
    DIR *pDir;
    pDir = opendir ((DIR_PATH).c_str());
    if (pDir == NULL)
    {
        // Creating a directory 
        if (mkdir((DIR_PATH).c_str(), 0777) == -1){
            cerr << "Error: " << strerror(errno) << endl;
            exit(1);
        } 
        if (mkdir((DIR_PATH+"/default").c_str(), 0755) == -1){
            cerr << "Error: " << strerror(errno) << endl;
            exit(1);
        }
        if (mkdir((DIR_PATH+"/allow").c_str(), 0755) == -1){
            cerr << "Error: " << strerror(errno) << endl;
            exit(1);
        }
        if (mkdir((DIR_PATH+"/deny").c_str(), 0755) == -1){
            cerr << "Error: " << strerror(errno) << endl;
            exit(1);
        }
        else
            printf("Directory created\n");    
    }
    (void) closedir (pDir);

    if (1) {
        string full_path = DIR_PATH + "public" + ".key";
        ifstream file(full_path);
        if (! file.is_open()){
            printf("Creating public key file\n");
            ofstream file(full_path);
            file.write((char *) pk, crypto_sign_PUBLICKEYBYTES);
        } else{
            printf("Public key for this user already exists!\n");
            exit(1);
        }
    }
    if (1) {
        string full_path = DIR_PATH + "private" + ".key";
        ifstream file(full_path);
        if (! file.is_open()){
            printf("Creating private key file\n");
            ofstream file(full_path);
            file.write((char *) sk, crypto_sign_SECRETKEYBYTES);
        } else{
            cout << "Secret key for this user already exists!\n";
            exit(1);
        }
    }
    if (1) {
        string full_path = DIR_PATH + "/udf.signed";
        ifstream pass_file(full_path);
        if(! pass_file.is_open()){
            printf("Signing file\n");
            ofstream file(full_path);
            file.write((char *) signed_message, signed_message_len);
        }
        else{
            printf("Signed file already exists!\n"); 
        }
    }
}