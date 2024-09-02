#!/usr/bin/env bash

set -e

if [ "$#" -ne 6 ]
then
    echo "USAGE: $0 ./my-list-of-files.txt <type> <csv-out-dir> <pic-out-dir> <QUBOdir> <orig-dir>"
    exit 1
fi

shopt -s extglob
if [[ $2 != @(ibm|dwave|quera) ]]; then
    echo "$2: wrong log type (ibm / dwave /quera expected)"
    exit 1
fi

files=`cat $1`

for file in $files; do
    instf=$(basename $file)
    echo "Processing $instf..."
    python -m post_processing.logparser extract_samples $2 $file $3/$instf.sample.csv $5 $6 && \
        Rscript post_processing/draw_samples.R -i $3/$instf.sample.csv -o $4/$instf.hist -s ./run_logs/solutions.csv
done
