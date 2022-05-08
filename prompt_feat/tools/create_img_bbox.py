import json
from PIL import Image
import json
import os
import sys
from oscar.utils.tsv_file import TSVFile
import json, pickle
import glob
import torch
import numpy as np

name = sys.argv[1]
tsvfile = TSVFile("output/X152C5_test/inference/vinvl_vg_x152c4/predictions.tsv")
img_infos = {}
for i in range(len(tsvfile)):
    img_key = tsvfile.seek(i)[0]
    data = json.loads(tsvfile.seek(i)[1])
    objs = data["objects"]
    bbox = [x['rect'] for x in objs]
    bbox = np.asarray(bbox, dtype=np.float32)
    img_infos[img_key] = bbox

pickle.dump(img_infos, open(os.path.dirname(name) + "/bbox.pk", "wb"))