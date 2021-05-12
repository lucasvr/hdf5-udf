
Configuration
=============

Configuration of trust profiles is made by adjusting the JSON files under the
corresponding trust directory:

- ``~/.config/hdf5-udf/deny/deny.json``: UDFs signed by previously unseen keys
  are associated with this profile.
- ``~/.config/hdf5-udf/deny/default.json``: settings for reasonably trusted keys
- ``~/.config/hdf5-udf/deny/allow.json``: settings for UDFs signed by trusted keys

The JSON file comprises the following keys:

- ``"sandbox"`` (boolean): set to `false` if you don't want to enforce a sandbox
  to UDFs associated with this profile, `true` otherwise.
- ``"syscalls"``: array of `key-value` objects describing the system call names
  and the conditions to allow UDFs to run them.
- ``"filesystem"``: array of `key-value` objects describing the paths an UDF can
  access on the filesystem.


Allowing system calls
---------------------

By default, all system calls are disallowed from being called by an UDF. The
``"syscalls"`` JSON array must explicitly state each system call a UDF can
call. There are two syntaxes to state so:

1. Allow a named system call to execute regardless of the arguments provided
   by the user:

.. code-block::

  {"syscall_name": true}


2. Allow a named system call to execute as long as the arguments match a given
   criteria. Examples:

.. code-block::

  # A rule for write(int fd, const void *buf, size_t count)
  {
    "write": {
      "arg": 0,               # First syscall argument (fd) ...
      "op": "equals",         # ... must be equal to ...
      "value": 1              # ... 1 (stdout)
    }
  }

  # A rule for open(const char *pathname, int flags)
  {
    "open": {                 
      "arg": 1,               # Second syscall argument (flags) ...
      "op": "masked_equals",  # ... when applied to bitmask ...
      "mask": "O_ACCMODE",    # ... O_ACCMODE ...
      "value": "O_RDONLY"     # ... must be equal to O_RDONLY
    }
  }

Mnemonic values such as ``"O_RDONLY"`` are automatically translated into
their numerical representation. They must be quoted as JSON strings.

Currently the two only possible values for ``op`` are ``equals`` and
``masked_equals``.

String-based filtering is not supported. Selection of which filesystem
paths a registered system call can access, however, is possible by
setting the ``"filesystem"`` array. See the next section for details.

Access to files and directories
-------------------------------

By default, access to any filesystem object is denied. The ``"filesystem"``
array can be used to state which parts of the filesystem can be accessed
and if they're available for programs opening objects in write mode or not.

The filesystem path component can be an absolute path or a string containing
wildcards (``*``). Two consecutive wildcards (``**``) can be used to recurse
into subdirectories. The supported open modes are ``ro`` for read-only access
and ``rw`` for both write-only and read-write requests.

Here are some examples. More settings can be found on the files shipped with
HDF5-UDF (e.g., ``~/.config/hdf5-udf/default/default.json``).

To allow access to any file, as long as the requested operation is read-only:

.. code-block::

  "filesystem": [
    {"/**": "ro"}
  ]

To allow access to Python packages:

.. code-block::

  "filesystem": [
    {"/**/python*/site-packages/**": "ro"}
  ]

To allow write access to /tmp:

.. code-block::

  "filesystem": [
    {"/tmp/**": "rw"}
  ]