#!/usr/bin/env sh

files=`cat ./instances/conv-test-qubo.list`

mkdir -p ./test-inst/QUBO

for file in $files; do
    cp ./instances/$file ./test-inst/QUBO/
done

files=`cat ./instances/conv-test-orig.list`

mkdir -p ./test-inst/orig

for file in $files; do
    cp ./instances/$file ./test-inst/orig/
done
