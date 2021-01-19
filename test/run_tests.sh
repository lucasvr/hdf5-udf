#!/bin/bash

# High-level regression test suite for HDF5-UDF.

set -e

# Logging (GoboLinux style)
colorGray="\033[1;30m"
colorBoldBlue="\033[1;34m"
colorBrown="\033[33m"
colorYellow="\033[1;33m"
colorBoldGreen="\033[1;32m"
colorBoldRed="\033[1;31m"
colorCyan="\033[36m"
colorBoldCyan="\033[1;36m"
colorRedWhite="\033[41;37m"
colorNormal="\033[0m"
colorBold="${colorYellow}"
colorOff="${colorNormal}"

function Log_Function() {
    local message="$1"
    local color="$2"
    local testName="$3"
    if [ "$testName" ]
    then echo -e "${colorGray}${testName}:${colorNormal} ${color}$message${colorNormal}"
    else echo -e "${color}$message${colorNormal}"
    fi
}

function Die() {
    local message="$1"
    local test_info="$2"
    Log_Function "$message" "${colorBoldRed}" "$test_info"
    Log_Function "Please check $RESULT_STDOUT and $RESULT_H5DUMP for details" "${colorBoldRed}" "$test_info"
    exit 1
}

function Log_Normal() {
    local message="$1"
    local test_info="$2"
    Log_Function "$message" "${colorCyan}" "$test_info"
}

# Helpers
function Read_Dataset() {
    local dataset="$1"
    local input_file="$2"
    h5dump -d "$dataset" -o "$RESULT_H5DUMP" -O "$input_file"
}

function CheckDiff() {
    local generated="$1"
    local reference="$2"
    if ! numdiff -q -a 1.0000000000e-6 "$reference" "$generated"
    then
        # Files are different. Show differences with 'diff'
        diff -up "$reference" "$generated"
    fi
}

function Run_Test() {
    local entry="$1"
    local test_name="$(echo $entry | awk {'print $1'})"
    local file_name="$(echo $entry | awk {'print $2'})"
    local params="$(echo $entry | awk {'print $3'})"
    local var_name="$(echo $params | cut -d: -f1)"

    if [ "$params" = "$var_name" ]
    then
        # Let hdf5-udf determine the dynamic dataset's name, resolution and data type
        params=
    fi

    for backend in "cpp" "py" "lua"
    do
        [ ! -e "${file_name}.${backend}" ] && continue

        # Adjust variable and dataset names so they're named after the backend being tested
        dataset_name="${var_name}.${backend}"
        if [ "$params" ]
        then dataset_info="$dataset_name:$(echo $params | cut -d: -f2-)"
        else dataset_info=
        fi

        test_info="${test_name} (${backend})"
        Log_Normal "Writing UDF" "${test_info}"
        hdf5-udf ${test_name}.h5 ${file_name}.${backend} ${dataset_info}

        validated=0
        Log_Normal "Reading UDF" "${test_info}"
        Read_Dataset /${dataset_name} ${test_name}.h5 > $RESULT_STDOUT
        if [ -e ${test_name}.stdout.${backend} ]
        then
            CheckDiff $RESULT_STDOUT ${test_name}.stdout.${backend} || Die "Validation failed" "${test_info}"
            validated=1
        elif [ -e ${test_name}.stdout ]
        then
            CheckDiff $RESULT_STDOUT ${test_name}.stdout || Die "Validation failed" "${test_info}"
            validated=1
        fi
        if [ -e ${test_name}.h5dump ]
        then
            diff -up $RESULT_H5DUMP ${test_name}.h5dump || Die "Validation failed" "${test_info}"
            validated=1
        fi
        if [ $validated = 0 ]
        then
            Die "No reference data available to validate test" "${test_info}"
        fi
        rm -f $RESULT_STDOUT $RESULT_H5DUMP
        Log_Normal "Successful validation" "${test_info}"
    done
    rm -f ${test_name}.h5
}

# Operation
export HDF5_PLUGIN_PATH=../src
export PATH=../src:$PATH
RESULT_STDOUT=stdout.txt
RESULT_H5DUMP=h5dump.txt

make -C ../examples files
cp -a ../examples/example-*.h5 .

# Notes:
# File names in the 'tests' array are given without their extension. The dynamic
# dataset may be given by its name alone (in which case hdf5-udf will determine
# its dimensions and data type) or the colon-separated name:resolution:type combo.
#
# To allow reuse of HDF5 files, each backend operates on dynamic datasets whose
# names are suffixed after that backend. For instance, 'test.cpp' will work on a
# dynamic dataset named 'VirtualDataset.cpp', 'test.lua' on 'VirtualDataset.lua'
# and so on. When adding new entries to the 'tests' array, please omit that suffix;
# the Run_Test() function will take care of putting it back when it's needed.

Log_Normal
Log_Normal "****************************"
Log_Normal "*** Single dataset tests ***"
Log_Normal "****************************"
Log_Normal
tests=(
    # HDF5 file             UDF file       Dynamic dataset
    "example-simple_vector  simple_vector  Simple:1500:float"
    "example-sine_wave      sine_wave      SineWave:100x10:int32"
)
for entry in "${tests[@]}"; do Run_Test "$entry"; done

Log_Normal
Log_Normal "*******************************"
Log_Normal "*** Multiple datasets tests ***"
Log_Normal "*******************************"
Log_Normal
tests=(
    # HDF5 file            UDF file             Dynamic dataset
    "example-add_datasets  test-multi-datasets  VirtualDataset"
)
for entry in "${tests[@]}"; do Run_Test "$entry"; done

Log_Normal
Log_Normal "********************"
Log_Normal "*** String tests ***"
Log_Normal "********************"
Log_Normal
tests=(
    # HDF5 file            UDF file             Dynamic dataset
    "example-string        test-string          Temperature:1000:double"
    #"example-varstring     test-string          Temperature:1000:double"
    #"example-multistring   test-multistring     Temperature:1000:double"
    "example-string2       test-string-output   RollingStone:405:string"
)
# Create a copy of example-string.h5, as the file gets removed after the
# first test finishes its execution. The copy is used for test-string-output.
cp example-string.h5 example-string2.h5
for entry in "${tests[@]}"; do Run_Test "$entry"; done

Log_Normal
Log_Normal "**********************"
Log_Normal "*** Compound tests ***"
Log_Normal "**********************"
Log_Normal
tests=(
    # HDF5 file                        UDF file                  Dynamic dataset
    "example-compound-nostring_simple  test-compound-nostring    Temperature:1000:double"
    "example-compound-nostring_mixed   test-compound-nostring    Temperature:1000:double"
    "example-compound-varstring_simple test-compound-string      Temperature:1000:double"
    "example-compound-varstring_mixed  test-compound-string      Temperature:1000:double"
    "example-compound-string_simple    test-compound-string      Temperature:1000:double"
    "example-compound-string_mixed     test-compound-string      Temperature:1000:double"
    "example-compound-varstring_mixed_plus_string test-compound-plus-string Temperature:1000:double"
)
for entry in "${tests[@]}"; do Run_Test "$entry"; done

rm -f -- *.h5