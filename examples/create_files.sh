#!/bin/bash

if [ $# != 3 ]
then
   echo "Syntax: $0 <SIZE> <CONTIGUOUS_FILE> <CHUNKED_FILE>"
   exit 1
fi

SIZE=$1
H5FILE=$2
H5FILE_CHUNKED=$3

#SNAPPY_USE_CUDA=0 ../build/createh5 --count=2 --dims=${SIZE},${SIZE} --compress=snappy-cuda --cdims=${SIZE},100 --out=${H5FILE_CHUNKED}
#SNAPPY_USE_CUDA=0 ../build/createh5 --count=2 --dims=${SIZE},${SIZE} --out=${H5FILE}
for ext in lua cpp py cu
do
#    hdf5-udf ${H5FILE} example-add_datasets.${ext}
#    hdf5-udf ${H5FILE_CHUNKED} example-add_datasets.${ext}
    hdf5-udf ${H5FILE} no-op.${ext}
    hdf5-udf ${H5FILE_CHUNKED} no-op.${ext}
done
