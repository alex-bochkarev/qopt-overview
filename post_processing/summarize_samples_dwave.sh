#!/usr/bin/env sh

python -m post_processing.logparser extract_all_samples dwave ./run_logs/dwave ./run_logs/dwave_samples

outf=./run_logs/dwave_samples/samples.csv
head -1 $(ls ./run_logs/dwave_samples/*.csv | shuf -n 1) > $outf
tail -qn +2 $(ls ./run_logs/dwave_samples/*.sample.csv) >> $outf
