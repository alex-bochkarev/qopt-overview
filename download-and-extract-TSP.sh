#!/usr/bin/env bash

wget -nd -r -P ./download/ -l1 -np "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"  -A "*.tsp.gz"
gunzip -f ./download/*.gz
