#!/usr/bin/env bash

mkdir -p ./figures/compressed

for pngfile in `ls ./figures/*.png`; do
    zopflipng $pngfile "./figures/compressed/$(basename $pngfile)"
done
