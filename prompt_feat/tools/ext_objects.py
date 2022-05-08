import os
import time
import json
import logging
import random
import glob
import base64
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from oscar.utils.tsv_file import TSVFile
from oscar.utils.misc import load_from_yaml_file
from oscar.utils.iou import computeIoU
import random
import sys


def load_image_features(feat_tsv, img_idx):
    img_name, feat_str = feat_tsv.seek(img_idx)
    feat_info = json.loads(feat_str)
    boxlist = feat_info["objects"]

    # od_labels
    boxlist_labels = [o['class'] for o in boxlist]
    od_labels = boxlist_labels
    # bboxes
    boxlist_rects = [o['rect'] for o in boxlist]
    rects = boxlist_rects
    return img_name, od_labels, None, rects

path = sys.argv[1]
feat_tsv = TSVFile(path)
dic = {}
save_path = os.path.join(sys.argv[2], "objects.json") if len(sys.argv) >= 3 else "objects.json"
for i in tqdm(range(len(feat_tsv))):
    img_name, od_labels, im_feats, rects = load_image_features(feat_tsv, i)
    dic[img_name] = list(zip(od_labels, rects))
json.dump(dic, open(save_path, "w"))


