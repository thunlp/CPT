# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import cv2
import math
import json, pickle
from PIL import Image
from maskrcnn_benchmark.structures.bounding_box import BoxList
import os.path as op
import numpy as np

from maskrcnn_benchmark.structures.tsv_file import TSVFile
import glob
import os
from PIL import Image, ImageDraw
import random


class Dset(object):
    def __init__(self):
        self.image_root = "/data_local/zhangao/data/coco/train2014"
        self.anns = json.load(open("data/refcoco/split/finetune_refcoco+_train.json")) # anns = [{}]
        self.anns = [k for k in self.anns if k['id'] == 197098]

        # load det file
        self.det_dic = json.load(open("data/refcoco/gts/refcoco+/dets.json")) # det_dic = {image_id: [box1, box2...]}

        self.colors = {'red': (250, 0, 45, 120), 'green': (100, 250, 0, 120), 'blue': (0, 40, 250, 120)}
        self.index = 0

    def __len__(self):
        return len(self.anns)

    def get_img_info(self, idx):
        data = self.anns[idx]
        return {'width': data['width'], 'height': data['height']}

    def get_img_id(self, name):
        name = name.replace(".jpg", "").split("_")[-1]
        return str(int(name))

    def __getitem__(self, idx):
        ann = self.anns[idx]
        caption = ann['caption']

        img_name = ann['file_name']
        img_path = os.path.join(self.image_root, img_name)
        img = Image.open(img_path).convert("RGB")
        img_size = img.size

        imid = self.get_img_id(img_name)
        dets = self.det_dic[imid]
        dets = [d['box'] for d in dets]
        size_iou_list = [box[-1]*box[-2]/float(img_size[0]*img_size[1]) for box in dets]
        dets = [box for box, iou in zip(dets, size_iou_list) if iou>0.1]

        gt_box = ann['bbox']
        iou_list = [self.computeIoU(box, gt_box) for box in dets]

        if len(iou_list) < 3:
            return None, None
        idxs = np.argsort(iou_list,)[::-1]
        tmp = [dets[idxs[0]]]
        tmp.append(dets[idxs[-1]])
        tmp.append(dets[idxs[-2]])
        dets = tmp

        dets = np.asarray(dets)
        dets[:, 2] = dets[:, 0] + dets[:, 2] - 1
        dets[:, 3] = dets[:, 1] + dets[:, 3] - 1
        dets = dets.tolist()


        target = BoxList(dets, img_size, mode="xyxy")
        p = self.draw_rectangles(img, target, list(self.colors.values()), imid=ann['id'], caption=caption)
        return caption, p

    def draw_rectangles(self, img, target, color_set, imid=None, caption=None):
        draw = ImageDraw.Draw(img)
        for i, (cname, color) in enumerate(self.colors.items()):
            box = target.bbox[i]
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            draw.rectangle(((x1, y1), (x2, y2)), outline="#32CD32", width=3)
        self.index += 1
        path = "/data_local/zhangao/codes/prompt_feat/tmp/imgs"
        path = os.path.join(path, str(imid))
        if not os.path.exists(path):
            os.mkdir(path)
        impath = os.path.join(path, str(self.index) + ".jpg")
        img.save(impath)
        with open(os.path.join(path, "caption.txt"), "w") as f:
            f.write(caption)
            f.flush()
        return path

    def computeIoU(self, box1, box2):
        # each box is of [x1, y1, w, h]
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[0] + box1[2] - 1, box2[0] + box2[2] - 1)
        inter_y2 = min(box1[1] + box1[3] - 1, box2[1] + box2[3] - 1)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
        else:
            inter = 0
        union = box1[2] * box1[3] + box2[2] * box2[3] - inter
        return float(inter) / union

dset = Dset()
import random
random.seed(3)
dset[0]