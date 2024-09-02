#!/usr/bin/env sh

files=`ls ./run_logs/dwave_samples/*sample.csv`

for file in $files; do
    instf=$(basename $file)
    Rscript ./post_processing/draw_samples_dwave.R -i $file -o ./run_logs/dwave_samples/pics/$instf
done
