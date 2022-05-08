#!/bin/bash

rm -rf output/vg/test/inference/

for i in 0 1 2 3 4
do
        sh cmds/vg/_vg_test.sh 5 $i
done