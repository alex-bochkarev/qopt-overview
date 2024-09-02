#!/usr/bin/env bash

set -e

# USAGE: $0 ./list-of-csv-files.txt <pic-out-dir> <type>

files=`cat $1`

for file in $files; do
    instf=$(basename $file)
    echo "Processing '$instf' from '$file'..."
    Rscript post_processing/samples_fig.R -i $file -o $2/$instf.hist -s ./run_logs/solutions.csv -t $3
done
