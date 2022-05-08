from PIL import Image
import numpy as np
import os
import pickle
from tqdm import tqdm

bbox = {}
for r in tqdm(range(0, 256, 5)):
    for g in range(0, 256, 5):
        for b in range(0, 256, 5):
            im = np.zeros([128, 128, 3])
            im[:, :, 0] = r
            im[:, :, 1] = g
            im[:, :, 2] = b
            im = Image.fromarray(im.astype(np.uint8))
            name = "{}_{}_{}.jpg".format(r, g, b)
            im.save(os.path.join("data/all_color_imgs/", name))
            bbox[name] = np.asarray([[0., 0, 128, 128]])
pickle.dump(bbox, open("data/all_color_imgs/bbox.pk", "wb"))


