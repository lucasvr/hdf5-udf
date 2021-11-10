#!/bin/bash

if [ $# != 1 ]
then
   echo "Syntax: $0 <SIZE>"
   exit 1
fi

SIZE=$1
#RESULTS_DIR="results-sandbox"
RESULTS_DIR="results-no_sandbox"

H5FILE=/nvme/example-add_datasets-contiguous-${SIZE}.h5
H5FILE_CHUNKED=/nvme/example-add_datasets-chunked-${SIZE}.h5
OUTDIR=$PWD/$RESULTS_DIR/results-${SIZE}x${SIZE}-contiguous
OUTDIR_CHUNKED=$PWD/$RESULTS_DIR/results-${SIZE}x${SIZE}-chunked

clear_caches() {
   echo 3 | sudo dd of=/proc/sys/vm/drop_caches &> /dev/null
}

run() {
   local threads=$1
   local dataset=$2
   local outfile=$3

   mkdir -p $OUTDIR $OUTDIR_CHUNKED
   export OMP_NUM_THREADS=$threads

   clear_caches
   /usr/bin/time -a -v -o $OUTDIR/$outfile readh5 $H5FILE $dataset --quiet >> $OUTDIR/internal_${outfile}
   
   #clear_caches
   #/usr/bin/time -a -v -o $OUTDIR_CHUNKED/$outfile readh5 $H5FILE_CHUNKED $dataset --quiet >> $OUTDIR_CHUNKED/internal_${outfile}
}

#./create_files.sh $SIZE $H5FILE $H5FILE_CHUNKED

for i in $(seq 10)
do
   run 1 Noop-Cpp noop.cpp
   run 1 Noop-Lua noop.lua
   run 1 Noop-Py noop.py
   
   run 1 OneDep-Cpp one_dep.cpp
   run 1 OneDep-Lua one_dep.lua
   run 1 OneDep-Py one_dep.py
done
