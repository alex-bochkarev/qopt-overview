#!/usr/bin/env sh

echo "D-Wave problems:"
while read file; do
    exception=$(jq '.solver.exception' $file)
    echo "$(basename $file): $exception"
done <./run_logs/problems_dwave.list

echo "IBM problems:"
while read file; do
    exception=$(jq '.solver.exception' $file)
    echo "$(basename $file): $exception"
done <./run_logs/problems_ibmq.list
