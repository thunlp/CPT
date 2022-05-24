#!/bin/bash

rm -rf output/vg/val/inference/

for i in 0 1 2 3 4
do
        bash cmds/vg/_vg_val.sh 5 $i
done