#!/bin/bash

rm -rf output/vg/test/inference/

for((i=0;i<=14;i++));
do
    bash cmds/vg/_vg_test.sh 15 $i
done
#for i in 0 1 2 3 4 5 6 7 8 9
#do
#done