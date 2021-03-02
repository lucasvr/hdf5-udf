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
#ifdef ENABLE_SANDBOX
#include <seccomp.h>
#endif
#include <map>
#include <string>
#include <vector>
#include <filesystem>

#include "user_profile.h"
#include "sysdefs.h"
#include "base64.h"

namespace fs = std::filesystem;
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

Blob *SignatureHandler::extractPayload(
    const uint8_t *in,
    unsigned long long size_in,
    std::string public_key_base64)
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
    if (ret == GLOB_NOMATCH)
    {
        // Import public key into the 'deny' repository.
        // Any system calls attempted to be executed by UDFs
        // signed by this key will be rejected by HDF5-UDF.
        std::string public_key;
        macaron::Base64 base64;
        auto errmsg = base64.Decode(public_key_base64, public_key);
        if (errmsg.size() > 0)
        {
            fprintf(stderr, "Base64: %s\n", errmsg.c_str());
            return NULL;
        }

        // Guarantee a unique file name for the new file
        std::string public_path = configdir + "deny/XXXXXX.pub";
        int fd = mkstemps(&public_path[0], 4);
        if (fd < 0)
        {
            fprintf(stderr, "Failed to create file at %s/deny: %s\n",
                configdir.c_str(), strerror(errno));
            return NULL;
        }
        close(fd);

        createDirectoryTree();
        savePublicKey((uint8_t *) public_key.c_str(), public_path, true);

        // Re-run glob() and fall-through so the test below can check for
        // any error conditions
        ret = glob(public_keys.c_str(), GLOB_BRACE, globError, &globbuf);
    }

    // Catch errors other than GLOB_NOMATCH from our original call to glob()
    // as well as errors coming from our re-execution of glob() in the 'if'
    // branch above.
    if (ret != 0)
    {
        fprintf(stderr, "Error scanning %s: %s\n", public_keys.c_str(), globStrError(ret));
        return NULL;
    }

    for (size_t i=0; i<globbuf.gl_pathc; ++i)
    {
        // Read public key
        auto path = globbuf.gl_pathv[i];
        ifstream file(path, ios::in|ios::binary|ios::ate);
        if (! file.is_open())
        {
            fprintf(stderr, "Error opening %s: %s\n", path, strerror(errno));
            continue;
        }
        auto file_size = file.tellg();
        file.seekg(0, ios::beg);
        auto pub_key = new uint8_t[file_size];
        file.read((char *) pub_key, file_size);

        // Attempt to decrypt the input data with the public key read
        auto data = new uint8_t[size_in];
        auto size = size_in;
        if (crypto_sign_open(data, &size, in, size_in, pub_key) != 0)
        {
            const char *extra = i == globbuf.gl_pathc-1 ? "" : ", will try another key";
            fprintf(stderr, "Could not decode UDF with %s%s\n", path, extra);
            delete[] pub_key;
            delete[] data;
            continue;
        }

        // Create a new Blob object and set its public_key_path member.
        // This way, the caller can identify the associated configuration
        // file holding the seccomp rules for this UDF.
        auto blob = new Blob(data, size);
        blob->public_key_path = path;

        delete[] pub_key;
        globfree(&globbuf);
        return blob;
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

    macaron::Base64 base64;
    auto public_key_base64 = base64.Encode(public_key, crypto_sign_PUBLICKEYBYTES);
    return new Blob(signed_message, signed_message_len, public_key_base64);
}

bool SignatureHandler::getProfileRules(std::string public_key_path, json &rules)
{
    std::string public_key_dir = fs::path(public_key_path).parent_path();
    if (fs::equivalent(public_key_dir, configdir))
    {
        // Special case: public key comes from the very same user (and machine)
        // who's running the UDF. Cook a JSON that allows this UDF to run with
        // no restrictions.
        rules["sandbox"] = false;
        return true;
    }

    std::string base = fs::path(public_key_dir).filename();
    std::string rulefile = public_key_dir + "/" + base + ".json";
    ifstream file(rulefile);
    if (! file.is_open())
    {
        // No matching rule file exists. Cook a JSON that denies this UDF from
        // running any system calls.
        fprintf(stderr, "**** special case: no rules -> deny\n");
        rules["sandbox"] = true;
        return true;
    }

    // Deserialize JSON from input stream
    file >> rules;

    return validateProfileRules(rulefile, rules);
}

#define Fail(fmt...) do { \
    fprintf(stderr, "%s, %s: ", rulefile.c_str(), name.c_str()); \
    fprintf(stderr, fmt); \
    fprintf(stderr, "\n"); \
    return false; \
} while (0)

bool SignatureHandler::validateProfileRules(std::string rulefile, json &rules)
{
    if (! rules.contains("sandbox"))
    {
        fprintf(stderr, "%s: missing mandatory 'sandbox' element\n", rulefile.c_str());
        return false;
    }
    else if (! rules.contains("syscalls"))
    {
        // syscalls array is optional
        return true;
    }
    else if (! rules["syscalls"].is_array())
    {
        fprintf(stderr, "%s: 'syscalls' element must be an array\n", rulefile.c_str());
        return false;
    }

    for (auto &[k, v]: rules["syscalls"].items())
        for (auto &[name, rule]: v.items())
        {
            if (name.size() && name[0] == '#')
                continue;

#ifdef ENABLE_SANDBOX
            // Validate syscall name
            if (seccomp_syscall_resolve_name(name.c_str()) == __NR_SCMP_ERROR)
                Fail("failed to resolve syscall name '%s'", name.c_str());
#endif

            if (rule.is_boolean())
                continue;
            else if (! rule.is_object())
                Fail("rules must be given as boolean or JSON object");

            // Check that required rule elements are present
            const char *needed[] = {"arg", "op", "value", NULL};
            for (int i=0; needed[i]; ++i)
                if (! rule.contains(needed[i]))
                    Fail("missing rule element '%s'", needed[i]);

            // Check that the data types are correct
            if (! rule["arg"].is_number())
                Fail("rule element 'arg' must be a number");
            else if (! rule["op"].is_string())
                Fail("rule element 'op' must be a string");
            else if (! (rule["value"].is_string() || rule["value"].is_number()))
                Fail("rule element 'value' must be a string or number");

            // Check that opcode is valid
            auto rule_op = rule["op"].get<std::string>();
            if (rule_op.compare("equals") && rule_op.compare("is_set"))
                Fail("invalid value for opcode: '%s'", rule_op.c_str());

            // Check that any given mnemonics are supported
            if (rule["value"].is_string())
            {
                auto value = rule["value"].get<std::string>();
                auto it = sysdefs.find(value);
                if (it == sysdefs.end())
                    Fail("unrecognized mnemonic '%s'", value.c_str());

                // Overwrite original string-based data with its integer value
                rule["value"] = it->second;
            }
        }
    return true;
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

bool SignatureHandler::savePublicKey(uint8_t *public_key, std::string path, bool overwrite)
{
    struct stat statbuf;
    if (overwrite || stat(path.c_str(), &statbuf) != 0)
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