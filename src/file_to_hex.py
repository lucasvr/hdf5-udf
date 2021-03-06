#!/usr/bin/env python3

# Simple xxd-like clone to convert a set of input files into
# C arrays of unsigned chars. This script is used to embed the
# contents of udf_template.{cpp,lua,py} into libhdf5_udf so they
# don't have to be distributed and located at runtime.

import os, sys

if len(sys.argv) < 3:
    print("Syntax: {} <input_file(s)> <output_file>".format(sys.argv[0]))
    sys.exit(1)

input_files = sys.argv[1:-1]
output_file = sys.argv[-1]
c_decl, cols = "", 16

for input_file in input_files:
    # Convert input file into array of hex digits
    with open(input_file, "rb") as f:
        hexdata = [format(x, "#04x") for x in f.read()] + ['0x00']

    # Store the array of hex digits on a C array
    varname = os.path.basename(input_file).replace(".", "_")
    c_decl += "\nstatic unsigned char {}[] = {{\n".format(varname)
    for i in range(0, len(hexdata), cols):
        c_decl += ", ".join(hexdata[i:i+cols]) + ",\n"
    c_decl = c_decl[:-2] + "\n};\n"

# Save into output file
guard_name = "__{}".format(os.path.basename(output_file).replace(".", "_"))
with open(output_file, "w") as f:
    f.write("// Automatically generated by {}\n".format(os.path.basename(__file__)))
    f.write("#ifndef {}\n".format(guard_name))
    f.write("#define {}\n".format(guard_name))
    f.write(c_decl)
    f.write("\n#endif\n")