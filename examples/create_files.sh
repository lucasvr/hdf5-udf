#!/bin/bash

#set -e
#set -x

if [ $# != 3 ]
then
   echo "Syntax: $0 <SIZE> <CONTIGUOUS_FILE> <CHUNKED_FILE>"
   exit 1
fi

SIZE=$1
H5FILE=$2
H5FILE_CHUNKED=$3

rm -f -- "$H5FILE" "$H5FILE_CHUNKED"

SNAPPY_USE_CUDA=0 ../build/createh5 --count=2 --dims=${SIZE},${SIZE} --compress=snappy-cuda --cdims=${SIZE},100 --out=${H5FILE_CHUNKED}
SNAPPY_USE_CUDA=0 ../build/createh5 --count=2 --dims=${SIZE},${SIZE} --out=${H5FILE}

for ext in lua cpp py cu
do
    echo "Dataset size $SIZE, backend $ext - contiguous"
    hdf5-udf ${H5FILE} example-add_datasets.${ext}
    
    echo "Dataset size $SIZE, backend $ext - chunked"
    hdf5-udf ${H5FILE_CHUNKED} example-add_datasets.${ext}

    #dset_name="Noop-$(echo $ext | sed -e 's,[a-z],\U&,1')"
    #hdf5-udf ${H5FILE} no-op.${ext} ${dset_name}:${SIZE}:int32
    #hdf5-udf ${H5FILE_CHUNKED} no-op.${ext} ${dset_name}:${SIZE}:int32
done
