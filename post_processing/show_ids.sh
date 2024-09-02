#!/usr/bin/env sh

files=`ls ./run_logs/IBM_v2/*.json`

for file in $files; do
    echo "$file → `jq .instance_id $file`"
done
