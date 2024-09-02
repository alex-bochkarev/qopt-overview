#!/usr/bin/env sh

set -e

if [ "$#" -ne 3 ]
then
    echo "USAGE: $0 ./my-list-of-files.txt <csv-out-dir> <pic-out-dir>"
    exit 1
fi

files=`cat $1`

for file in $files; do
    instf=$(basename $file)
    echo "Processing $instf"
    python -m post_processing.logparser extract_convergence_data ibm $file $2/$instf.conv.csv && \
        Rscript ./post_processing/convergence.R -i "$2/$instf.conv.csv" -o "$3/$instf.conv.png"
done
