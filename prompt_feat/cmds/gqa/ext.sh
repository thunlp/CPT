#!/bin/bash

rm -rf output/gqa/gqa_img_feats

for((i=0;i<=14;i++));
do
  echo $i;
  python tools/cnt.py $i
  bash cmds/gqa/_ext.sh
done

mv output/gqa/gqa_img_feats/inference/vinvl_vg_x152c4/* output/gqa/gqa_img_feats/
rm -rf output/gqa/gqa_img_feats/inference/


python tools/ext_objects.py output/gqa/gqa_img_feats/predictions.tsv output/gqa/gqa_img_feats/
