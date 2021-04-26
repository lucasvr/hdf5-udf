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
#include <libgen.h>
#include <sys/utsname.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <pwd.h>
#ifdef ENABLE_SANDBOX
#include <seccomp.h>
#endif
#include <map>
#include <string>
#include <vector>
#include <iomanip>

#include "user_profile.h"
#include "sysdefs.h"
#include "base64.h"

using namespace std;

bool filesystem_path_exists(std::string path)
{
    struct stat statbuf;
    return stat(path.c_str(), &statbuf) < 0 && errno == ENOENT ? false : true;
}

bool filesystem_paths_equal(std::string a, std::string b)
{
    struct stat info_a, info_b;
    if (stat(a.c_str(), &info_a) < 0 || stat(b.c_str(), &info_b) < 0)
        return false;
    return info_a.st_dev == info_b.st_dev && info_a.st_ino == info_b.st_ino;
}

std::string filesystem_parentdir(std::string path)
{
    char tmp[path.size()+1];
    sprintf(tmp, "%s", path.c_str());
    return dirname(tmp);
}

std::string filesystem_basename(std::string path)
{
    char tmp[path.size()+1];
    sprintf(tmp, "%s", path.c_str());
    return basename(tmp);
}

bool get_raw_key(const json &pk, std::string &raw_key)
{
    macaron::Base64 base64;
    auto errmsg = base64.Decode(pk.get<std::string>(), raw_key);
    if (errmsg.size() > 0)
    {
        fprintf(stderr, "Base64: %s\n", errmsg.c_str());
        return false;
    }
    return true;
}

bool get_base64_key(uint8_t *raw_key, size_t size, std::string &base64_key)
{
    macaron::Base64 base64;
    base64_key = base64.Encode(raw_key, size);
    return true;
}

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
    json &signature,
    bool first_call)
{
    if (first_call && ! signature.contains("public_key"))
    {
        fprintf(stderr, "Error: signature does not contain a public_key\n");
        return NULL;
    }
    else if (first_call && sodium_init() == -1)
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
    if (first_call && ret == GLOB_NOMATCH)
    {
        // This is a brand new installation: no keys have been imported yet.
        if (importPublicKey(signature) == false)
            return NULL;
        return extractPayload(in, size_in, signature, false);
    }
    else if (ret != 0)
    {
        fprintf(stderr, "Error scanning %s: %s\n", public_keys.c_str(), globStrError(ret));
        return NULL;
    }

    Blob *blob = findPublicKey(in, size_in, globbuf);
    globfree(&globbuf);
    if (first_call && blob == NULL)
    {
        // We have never seen this key before. Import it and retry.
        if (importPublicKey(signature) == false)
            return NULL;
        return extractPayload(in, size_in, signature, false);
    }

    return blob;
}

Blob *SignatureHandler::findPublicKey(
    const uint8_t *in,
    unsigned long long size_in,
    glob_t &globbuf)
{
    for (size_t i=0; i<globbuf.gl_pathc; ++i)
    {
        // Read public key
        auto path = globbuf.gl_pathv[i];
        ifstream file(path);
        if (! file.is_open())
        {
            fprintf(stderr, "Error opening %s: %s\n", path, strerror(errno));
            continue;
        }
        json candidate;
        file >> candidate;
        file.close();

        if (! candidate.contains("public_key"))
        {
            fprintf(stderr, "Error: %s does not contain a public_key entry\n", path);
            continue;
        }

        std::string public_key;
        if (get_raw_key(candidate["public_key"], public_key) == false)
        {
            fprintf(stderr, "Error: failed to parse public key from file %s\n", path);
            continue;
        }

        // Attempt to decrypt the input data with the public key read
        auto data = new uint8_t[size_in];
        auto size = size_in;
        if (crypto_sign_open(data, &size, in, size_in, (unsigned char *) &public_key[0]) != 0)
        {
            delete[] data;
            continue;
        }

        // Create a new Blob object and set its public_key_path member.
        // This way, the caller can identify the associated configuration
        // file holding the seccomp rules for this UDF.
        auto blob = new Blob(data, size);
        blob->public_key_path = path;
        return blob;
    }

    return NULL;
}

bool SignatureHandler::importPublicKey(json &signature)
{
    if (! createDirectoryTree())
    {
        fprintf(stderr, "Unable to import public key into local repository\n");
        return false;
    }

    // Import public key into the 'deny' repository.
    // Any system calls attempted to be executed by UDFs
    // signed by this key will be rejected by HDF5-UDF.
    std::string email = signature.contains("email") ?
        signature["email"].get<std::string>() : "unknown";

    // Find a valid name slot
    for (int i=0; i<1024; ++i)
    {
        std::stringstream ss;
        ss << configdir << "deny/" << email;
        if (i > 0)
            ss << "." << i;
        ss << ".pub";

        struct stat statbuf;
        if (stat(ss.str().c_str(), &statbuf) != 0)
            return saveFile(signature, ss.str(), true);
    }

    return false;
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
    json public_json;

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
        ifstream file(path);
        if (! file.is_open())
        {
            fprintf(stderr, "Error opening private key %s\n", path.c_str());
            return NULL;
        }
        json private_json;
        file >> private_json;
        file.close();

        if (! private_json.contains("private_key"))
        {
            fprintf(stderr, "Error: %s does not contain a private_key entry\n", path.c_str());
            return NULL;
        }

        std::string secret_key_str;
        if (get_raw_key(private_json["private_key"], secret_key_str) == false)
        {
            fprintf(stderr, "Error: failed to parse private key from file %s\n", path.c_str());
            return NULL;
        }
        else if (secret_key_str.size() != crypto_sign_SECRETKEYBYTES)
        {
            fprintf(stderr, "Error: private key %s has %d bytes, expected %u instead\n",
                path.c_str(), static_cast<int>(secret_key_str.size()), crypto_sign_SECRETKEYBYTES);
            return NULL;
        }
        memcpy(secret_key, &secret_key_str[0], crypto_sign_SECRETKEYBYTES);

        // Extract the public key from the secret key
        if (crypto_sign_ed25519_sk_to_pk(public_key, secret_key) != 0)
        {
            fprintf(stderr, "Error extracting public key from secret %s\n", path.c_str());
            return NULL;
        }

        // Attempt to read key metadata
        std::string public_path = path.substr(0, path.find_last_of('.')) + ".pub";
        ifstream public_file(public_path);
        if (public_file.is_open())
            public_file >> public_json;
    }
    else if (ret == GLOB_NOMATCH)
    {
        // Create new public and private keys
        if (crypto_sign_keypair(public_key, secret_key))
        {
            fprintf(stderr, "Error generating public and secret keys\n");
            return NULL;
        }

        std::string base64_public_key, base64_private_key;
        if (get_base64_key(public_key, crypto_sign_PUBLICKEYBYTES, base64_public_key) == false)
        {
            fprintf(stderr, "Failed to convert public key to base64\n");
            return NULL;
        }
        if (get_base64_key(secret_key, crypto_sign_SECRETKEYBYTES, base64_private_key) == false)
        {
            fprintf(stderr, "Failed to convert private key to base64\n");
            return NULL;
        }

        if (! createDirectoryTree())
        {
            fprintf(stderr, "Unable to save generated public and secret keys\n");
            return NULL;
        }

        // Create metadata
        struct utsname uts;
        memset(&uts, 0, sizeof(uts));
        uname(&uts);
        auto pw = getpwuid(getuid());
        json private_json;

        private_json["private_key"] = base64_private_key;
        public_json["name"] = pw ? (strlen(pw->pw_gecos) ? pw->pw_gecos : pw->pw_name) : "Unknown";
        public_json["email"] = pw ? std::string(pw->pw_name) + "@" + std::string(uts.nodename) : "user@email";
        public_json["public_key"] = base64_public_key;

        // Save both keys to disk
        std::string private_path = configdir + (pw ? pw->pw_name : "my") + ".priv";
        std::string public_path = configdir + (pw ? pw->pw_name : "my") + ".pub";

        saveFile(private_json, private_path);
        saveFile(public_json, public_path);
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

    auto blob = new Blob(signed_message, signed_message_len, public_json["public_key"]);
    blob->metadata = public_json;
    return blob;
}

bool SignatureHandler::getProfileRules(std::string public_key_path, json &rules)
{
    std::string public_key_dir = filesystem_parentdir(public_key_path);
    if (filesystem_paths_equal(public_key_dir, configdir))
    {
        // Special case: public key comes from the very same user (and machine)
        // who's running the UDF. Cook a JSON that allows this UDF to run with
        // no restrictions.
        rules["sandbox"] = false;
        return true;
    }

    // This function is expected to be called from the I/O filter succeeding
    // a call to extractPayload(). Since that function performs the extraction
    // of the public key and stores it under the 'deny' profile (iff that key
    // was not previously associated with another profile), we really don't
    // expect the next operation to fail.
    std::string base = filesystem_basename(public_key_dir);
    std::string rulefile = public_key_dir + "/" + base + ".json";
    ifstream file(rulefile);
    if (! file.is_open())
    {
        fprintf(stderr, "Error: could not open profile rule file %s\n", rulefile.c_str());
        return false;
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

    for (auto &element: rules["syscalls"].items())
        for (auto &syscall_element: element.value().items())
        {
            auto name = syscall_element.key();
            auto rule = syscall_element.value();
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
            }
        }
    return true;
}

#define JSON_OBJ(k,v) json::object({{k, v}})

bool SignatureHandler::createDirectoryTree()
{
    // Create config directory tree
    std::vector<std::string> dirs;
    dirs.push_back(configdir);
    dirs.push_back(configdir + "default");
    dirs.push_back(configdir + "allow");
    dirs.push_back(configdir + "deny");

    for (auto &dir: dirs)
        if (! filesystem_path_exists(dir))
            if (mkdir(dir.c_str(), 0755) < 0)
            {
                fprintf(stderr, "Error creating directory %s: %s\n", dir.c_str(), strerror(errno));
                return false;
            }

    // Create default config files
    json default_cfg, allow_cfg, deny_cfg;

    // "Allow": don't enforce sandboxing rules
    allow_cfg["#"] = "Run the UDF in a sandbox?";
    allow_cfg["sandbox"] = false;
    if (! filesystem_path_exists(configdir + "allow/allow.json"))
        std::ofstream(configdir + "allow/allow.json") << std::setw(4) << allow_cfg << std::endl;

    // "Deny": don't let the UDF run any syscalls, except for munmap and write(stdout/err)
    deny_cfg["#"] = "Run the UDF in a sandbox?";
    deny_cfg["sandbox"] = true;
    deny_cfg["syscalls"] = json::array({
        JSON_OBJ("munmap", true),
        JSON_OBJ("write", json::object({{"arg", 0}, {"op", "equals"}, {"value", 1}})),
        JSON_OBJ("write", json::object({{"arg", 0}, {"op", "equals"}, {"value", 2}})),
    });
    if (! filesystem_path_exists(configdir + "deny/deny.json"))
        std::ofstream(configdir + "deny/deny.json") << std::setw(4) << deny_cfg << std::endl;

    // "Default": let common-sense system calls execute.
    // Note that some syscalls are explicitly disabled, even though that's the default,
    // so that users can easily turn them on if so they wish.
    default_cfg["#"] = "Run the UDF in a sandbox?";
    default_cfg["sandbox"] = true;
    default_cfg["syscalls"] = json::array({
        JSON_OBJ("# Memory management", ""),
        JSON_OBJ("brk", true),
        JSON_OBJ("futex", true),
        JSON_OBJ("mprotect", true),
        JSON_OBJ("mmap", true),
        JSON_OBJ("mmap2", true),
        JSON_OBJ("munmap", true),

        JSON_OBJ("# Process management", ""),
        JSON_OBJ("exit_group", true),

        JSON_OBJ("# Terminal-related", ""),
        JSON_OBJ("ioctl", json::object({{"arg", 1}, {"op", "equals"}, {"value", "TCGETS"}})),
        JSON_OBJ("ioctl", json::object({{"arg", 1}, {"op", "equals"}, {"value", "TIOCGWINSZ"}})),

        JSON_OBJ("# System information", ""),
        JSON_OBJ("uname", true),

        JSON_OBJ("# Sockets-related system calls", ""),
        JSON_OBJ("ioctl", json::object({{"arg", 1}, {"op", "equals"}, {"value", "FIONREAD"}})),

        JSON_OBJ("# File descriptor", ""),
        JSON_OBJ("open", json::object({{"arg", 1}, {"op", "is_set"}, {"value", "O_RDONLY"}})),
        JSON_OBJ("openat", json::object({{"arg", 2}, {"op", "is_set"}, {"value", "O_RDONLY"}})),
        JSON_OBJ("close", true),
        JSON_OBJ("read", false),
        JSON_OBJ("write", json::object({{"arg", 0}, {"op", "equals"}, {"value", 1}})),
        JSON_OBJ("write", json::object({{"arg", 0}, {"op", "equals"}, {"value", 2}})),
        JSON_OBJ("fcntl", true),
        JSON_OBJ("fcntl64", true),
        JSON_OBJ("stat", true),
        JSON_OBJ("lstat", true),
        JSON_OBJ("fstat", true),
        JSON_OBJ("fstat64", true),
        JSON_OBJ("lseek", true),
        JSON_OBJ("_llseek", true)
    });
    if (! filesystem_path_exists(configdir + "default/default.json"))
        std::ofstream(configdir + "default/default.json") << std::setw(4) << default_cfg << std::endl;

    return true;
}

bool SignatureHandler::saveFile(const json &metadata, std::string path, bool overwrite)
{
    struct stat statbuf;
    if (overwrite || stat(path.c_str(), &statbuf) != 0)
    {
        ofstream file(path);
        file << std::setw(4) << metadata << std::endl;
    }
    return true;
}