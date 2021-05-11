# Security considerations

Trusting user-defined functions to execute arbitrary code on someone else's
computer is always complicated. All is fine until some malicious piece of code
decides to explore the filesystem and mess around with system resources and
sensitive data. This is such an important topic that several projects delegate
dedicated teams to reduce the attack surface of components that execute code
from external parties.

Because users cannot tell in advance what a certain user-defined function is
about to do, HDF5-UDF uses a few mechanisms to limit what system calls the
UDF can execute.

### Seccomp

We rely on **seccomp** to determine which system calls UDFs are allowed to
invoke -- the UDF process is terminated if it tries to run a function that
does not belong to the allow-list. The following image shows the overall
architecture of our seccomp-based sandboxing.

![](images/hdf5-udf-seccomp.png)

### Trust profiles

Starting with HDF5-UDF 2.0, a private and public key pair is automatically generated
and saved to the user's home directory (under `~/.config/hdf5-udf`) the first time
a dataset is created. The files are named after the currently logged user name:

- `~/.config/hdf5-udf/username.priv` private key
- `~/.config/hdf5-udf/username.pub`: public information: public key, email, and full
  name. The last two pieces of information are automatically assembled from `hostname`
  and `/etc/passwd`. Please review and adjust the file as you see fit.

A directory structure providing different `trust profiles` is also created. Inside
each profile directory exists a JSON file which states the system calls allowed to
be executed by members of that profile. Three profiles are created:

- **default**: a sane configuration that allows memory allocation, opening files in
  read-only mode, writing to `stdout` and `stderr`, and interfacing with the
  terminal device.
- **deny**: strict settings that simply allow writing to `stdout` and `stderr`.
- **allow**: poses no restrictions. The UDF is treated as a regular process with
  no special requirements.

![](images/profiles.png)

### Signing UDFs

UDFs are **automatically signed** at the time of their attachment to the HDF5 file.
The public key from `username.pub` and contact information from `username.meta`
are incorporated as metadata and saved next to the UDF bytecode in the HDF5 file.

### Associating UDFs with a trust profile

Self-signed UDFs are automatically placed on the `allow` profile. This means that
UDFs you create on your own machine will run, on that same machine, as a regular
process would.

HDF5 files with UDFs signed by a different user are automatically placed on the
`deny` profile: the public key is extracted from the metadata and saved as
`~/.config/hdf5-udf/deny/foo.pub`. In other words, when you receive a file from
an unknown party and load a UDF dataset, the bytecode will not be able to perform
any actions that require the execution of system calls (other than writing to
`stdout` and `stderr`).

It is possible to change the trust level by simply **moving that public key to a
different profile directory**. The next time a UDF signed by that key is read,
the seccomp rules associated with that profile will be enforced.

### Ptrace

It is possible to configure **seccomp** rules so that only system calls with
specific arguments are allowed to execute. The exception is that string-based arguments are not processed by **seccomp** filters. Consequently, we have to
combine **seccomp** with another mechanism to provide fine-grained control of 
path-based filesystem operations.

We use the **ptrace** interface to intercept system calls issued by the UDF.
Attempts to access the filesystem are then checked against the trust profile
associated with that UDF. Accesses that violate the list of configured paths
lead the UDF process to be terminated.