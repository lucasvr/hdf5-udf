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
#include <sys/types.h>
#include <sys/stat.h>
#include <map>
#include <sstream>
#include <string>
#include <iomanip>

#include "config.h"
#include "user_profile.h"
#include "file_search.h"
#include "sysdefs.h"
#include "base64.h"
#include "os.h"

using namespace std;

bool filesystem_path_exists(std::string path)
{
    struct stat statbuf;
    return stat(path.c_str(), &statbuf) < 0 && errno == ENOENT ? false : true;
}

bool filesystem_paths_equal(std::string a, std::string b)
{
    struct stat info_a, info_b;
    // MINGW64 doesn't seem to succeed calls to stat() if the given directory
    // ends on a backslash (neither on a regular slash). Sigh..
    if (a.size() && a.compare(a.size()-1, 1, "\\") == 0)
        a = a.substr(0, a.size()-1);
    if (b.size() && b.compare(b.size()-1, 1, "\\") == 0)
        b = b.substr(0, b.size()-1);
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

bool string_in_vector(const std::string &s, const std::vector<std::string> &v)
{
    for (auto &entry: v)
        if (s.compare(entry) == 0)
            return true;
    return false;
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
    std::vector<std::string> candidates;
    if (findByExtension(configdir, ".pub", candidates) == false)
        return NULL;
    else if (candidates.size() == 0)
    {
        // This is a brand new installation: no keys have been imported yet.
        if (importPublicKey(signature) == false)
            return NULL;
        return extractPayload(in, size_in, signature, false);
    }

    Blob *blob = findPublicKey(in, size_in, candidates);
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
    const std::vector<std::string> &candidates)
{
    for (auto &path: candidates)
    {
        // Read public key
        ifstream file(path);
        if (! file.is_open())
        {
            fprintf(stderr, "Error opening %s: %s\n", path.c_str(), strerror(errno));
            continue;
        }
        json candidate;
        try
        {
            file >> candidate;
            file.close();
        }
        catch (nlohmann::detail::parse_error& e)
        {
            fprintf(stderr, "Error parsing %s:\n%s\n", path.c_str(), e.what());
            continue;
        }

        if (! candidate.contains("public_key"))
        {
            fprintf(stderr, "Error: %s does not contain a public_key entry\n", path.c_str());
            continue;
        }

        std::string public_key;
        if (get_raw_key(candidate["public_key"], public_key) == false)
        {
            fprintf(stderr, "Error: failed to parse public key from file %s\n", path.c_str());
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
    // Note that we pick the first one returned by the scan.
    std::vector<std::string> candidates;
    if (findByExtension(configdir, ".priv", candidates, false) == false)
        return NULL;
    else if (candidates.size() > 0)
    {
        path = candidates[0];

        // Read private key from the file
        ifstream file(path);
        if (! file.is_open())
        {
            fprintf(stderr, "Error opening private key file %s\n", path.c_str());
            return NULL;
        }
        json private_json;
        try
        {
            file >> private_json;
            file.close();
        }
        catch (nlohmann::detail::parse_error& e)
        {
            fprintf(stderr, "Error parsing %s:\n%s\n", path.c_str(), e.what());
            return NULL;
        }

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

        // Read public key and metadata file
        std::string public_path = path.substr(0, path.find_last_of('.')) + ".pub";
        ifstream public_file(public_path);
        if (! public_file.is_open())
        {
            fprintf(stderr, "Error opening public key file %s\n",public_path.c_str());
            return NULL;
        }
        try
        {
            public_file >> public_json;
        }
        catch (nlohmann::detail::parse_error& e)
        {
            fprintf(stderr, "Error parsing %s:\n%s\n", path.c_str(), e.what());
            return NULL;
        }
    }
    else if (candidates.size() == 0)
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
        json private_json;
	std::string username, login, host;
	os::getUserInformation(username, login, host);

        private_json["private_key"] = base64_private_key;
        public_json["name"] = username;
        public_json["email"] = login + "@" + host;
        public_json["public_key"] = base64_public_key;

        // Save both keys to disk
        std::string private_path = configdir + login + ".priv";
        std::string public_path = configdir + login + ".pub";

        saveFile(private_json, private_path);
        saveFile(public_json, public_path);
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

#define JSON_OBJ(k,v) json::object({{k, v}})

bool SignatureHandler::getProfileRules(std::string public_key_path, json &rules)
{
    std::string public_key_dir = filesystem_parentdir(public_key_path);
    if (filesystem_paths_equal(public_key_dir, configdir))
    {
        // Special case: public key comes from the very same user (and machine)
        // who's running the UDF. Cook a JSON that allows this UDF to run with
        // no restrictions. Note that there's no privilege escalation here: the
        // UDF will run with the same permissions and capabilities the user
        // already has.
        rules["sandbox"] = false;
        rules["filesystem"] = json::array({JSON_OBJ("/**", "rw")});
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
    try
    {
        file >> rules;
    }
    catch (nlohmann::detail::parse_error& e)
    {
        fprintf(stderr, "Error parsing %s:\n%s\n", rulefile.c_str(), e.what());
        return false;
    }

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
    if (! validateSandbox(rulefile, rules))
        return false;
    else if (! validateSandbox(rulefile, rules))
        return false;
    else if ((! validateFilesystem(rulefile, rules)))
        return false;
    return true;
}

bool SignatureHandler::validateSandbox(std::string rulefile, json &rules)
{
    if (! rules.contains("sandbox"))
    {
        std::string name = "sandbox";
        Fail("element is mandatory");
    }
    return true;
}

bool SignatureHandler::validateSyscalls(std::string rulefile, json &rules)
{
    if (! rules.contains("syscalls"))
    {
        // syscalls array is optional
        return true;
    }
    else if (! rules["syscalls"].is_array())
    {
        std::string name = "syscalls";
        Fail("must be an array");
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
            if (os::syscallNameToNumber(name.c_str()) < 0)
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
            if (rule_op.compare("equals") && rule_op.compare("masked_equals"))
                Fail("invalid value for opcode: '%s'", rule_op.c_str());
            else if (rule_op.compare("masked_equals") == 0 && ! rule.contains("mask"))
                Fail("missing 'mask' element for opcode '%s'", rule_op.c_str());

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

bool SignatureHandler::validateFilesystem(std::string rulefile, json &rules)
{
    if (! rules.contains("filesystem"))
    {
        // filesystem array is optional
        return true;
    }
    else if (! rules["filesystem"].is_array())
    {
        std::string name = "filesystem";
        Fail("must be an array");
    }

    for (auto &element: rules["filesystem"].items())
        for (auto &fs_element: element.value().items())
        {
            auto name = fs_element.key();
            auto rule = fs_element.value();
            if (name.size() && name[0] == '#')
                continue;

            if (! rule.is_string())
                Fail("path access mode must be given by a string");

            auto access_mode = rule.get<std::string>();
            std::vector<std::string> modes = {"ro", "rw"};
            if (! string_in_vector(access_mode, modes))
                Fail("invalid access mode '%s'", access_mode.c_str());
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

    for (auto &dir: dirs)
        if (! filesystem_path_exists(dir))
            if (os::createDirectory(dir, 0755) == false)
            {
                fprintf(stderr, "Error creating directory %s: %s\n", dir.c_str(), strerror(errno));
                return false;
            }

    // Create default config files
    json default_cfg, allow_cfg, deny_cfg;

    // "Allow": don't enforce sandboxing rules
    allow_cfg["sandbox"] = false;
    allow_cfg["filesystem"] = json::array({JSON_OBJ("/**", "rw")});
    if (! filesystem_path_exists(configdir + "allow/allow.json"))
        std::ofstream(configdir + "allow/allow.json") << std::setw(4) << allow_cfg << std::endl;

    // "Deny": only allow fundamental system calls
    // No filesystem access is permitted.
    deny_cfg["sandbox"] = true;
    deny_cfg["syscalls"] = json::array({
        JSON_OBJ("# Memory management", ""),
        JSON_OBJ("brk", true),
        JSON_OBJ("futex", true),
        JSON_OBJ("mprotect", true),
        JSON_OBJ("mmap", true),
        JSON_OBJ("mmap2", true),
        JSON_OBJ("munmap", true),

        JSON_OBJ("# Process management", ""),
        JSON_OBJ("exit_group", true),
        JSON_OBJ("rt_sigprocmask", true),

        JSON_OBJ("# Write access to stdout/stderr", ""),
        JSON_OBJ("write", json::object({{"arg", 0}, {"op", "equals"}, {"value", 1}})),
        JSON_OBJ("write", json::object({{"arg", 0}, {"op", "equals"}, {"value", 2}}))
    });
    deny_cfg["filesystem"] = json::array();
    if (! filesystem_path_exists(configdir + "deny/deny.json"))
        std::ofstream(configdir + "deny/deny.json") << std::setw(4) << deny_cfg << std::endl;

    // "Default": let common-sense system calls execute.
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
        JSON_OBJ("rt_sigprocmask", true),

        JSON_OBJ("# Terminal-related", ""),
        JSON_OBJ("ioctl", json::object({{"arg", 1}, {"op", "equals"}, {"value", "TCGETS"}})),
        JSON_OBJ("ioctl", json::object({{"arg", 1}, {"op", "equals"}, {"value", "TIOCGWINSZ"}})),

        JSON_OBJ("# System information", ""),
        JSON_OBJ("uname", true),

        JSON_OBJ("# Sockets-related system calls", ""),
        JSON_OBJ("ioctl", json::object({{"arg", 1}, {"op", "equals"}, {"value", "FIONREAD"}})),

        JSON_OBJ("# File descriptor", ""),
        JSON_OBJ("open", json::object({
            {"arg", 1},
            {"op", "masked_equals"},
            {"mask", "O_ACCMODE"},
            {"value", "O_RDONLY"}
        })),
        JSON_OBJ("openat", json::object({
            {"arg", 2},
            {"op", "masked_equals"},
            {"mask", "O_ACCMODE"},
            {"value", "O_RDONLY"}
        })),
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

    default_cfg["filesystem"] = json::array({
        JSON_OBJ("/**/python*/site-packages/**", "ro"),
        JSON_OBJ("/**/libm*", "ro"),
        JSON_OBJ("/**/libdl*", "ro"),
        JSON_OBJ("/**/ld-linux*", "ro"),
        JSON_OBJ("/**/libssl*", "ro"),
        JSON_OBJ("/**/libcrypto*", "ro"),
        JSON_OBJ("/**/libpthread*", "ro"),
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
