#!/usr/bin/env bash

# make_sample_figures <INST_LIST_FILE> <QPU: dwave, quera, or ibm> <folder-prefix>
#
# (folder-prefix is ibm-sim or ibm-qpu for the respective
# devices, or quera/dwave otherwise)

function make_sample_figures () {
    echo "" > $1.logfiles
    for ID in `cat $1`; do
        LOGFILE=$(cat "./run_logs/summaries/$2_summary.csv" | grep ",$ID," | \
            awk -F ',' '{print $1}')
        echo "$LOGFILE" >> $1.logfiles
        echo "Extracting samples from $LOGFILE..."
        mkdir -p ./run_logs/$2/samples-csv
        SAMPLEFILE="./run_logs/$2/samples-csv/$(basename $LOGFILE).sample.csv"
        LOGTYPE=$(basename $2 -sim)
        LOGTYPE=$(basename $LOGTYPE -qpu)
        python -m post_processing.logparser extract_samples $LOGTYPE $LOGFILE $SAMPLEFILE \
            ./instances/QUBO ./instances/orig
        echo "Done. Making the figure..."
        Rscript post_processing/samples_fig.R -i $SAMPLEFILE \
            -o "./figures/$2-$ID.hist" \
            -s ./run_logs/classic_solutions/solutions_all.csv \
            -t $LOGTYPE
        echo "Done."
    done
}

# Actually making the figures
for QPU in dwave quera ibm-sim ibm-qpu; do
    make_sample_figures "./${QPU}_suppl_samples.ids" "${QPU}"
done
