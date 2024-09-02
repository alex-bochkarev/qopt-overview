#!/usr/bin/env bash

while read p; do
  echo "file: $p"
  jq ".solver.exception" $p
done < ./run_logs/ibm-qpu-failed.loglist
