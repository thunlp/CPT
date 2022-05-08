import json
from PIL import Image
import json
import os
import sys
import glob
name = sys.argv[1]
filenames = glob.glob(os.path.join(name, "*.jpg"))
# filenames = {os.path.basename(fname): fname for fname in filenames}

img_infos = {}
for i, fname in enumerate(filenames):
    img = Image.open(filenames[i]).convert("RGB")
    w, h = img.size[0], img.size[1]
    img_infos[os.path.basename(fname)] = {"width":w, "height":h}

json.dump(img_infos, open(os.path.dirname(name) + "/img_info"+".json", "w"))