# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import cv2
import math
import json, pickle
from PIL import Image
from maskrcnn_benchmark.structures.bounding_box import BoxList
import os.path as op
import numpy as np

from maskrcnn_benchmark.structures.tsv_file import TSVFile
from .utils.load_files import load_linelist_file, load_from_yaml_file
from .utils.load_files import find_file_path_in_yaml
from .utils.image_ops import img_from_base64
import glob
import os
from PIL import Image, ImageDraw
import random


class RefcocoDemoDataset(object):
    def __init__(self, yaml_file, transforms=None, **kwargs):
        self.cfg = load_from_yaml_file(yaml_file)
        self.root = self.cfg['ann_root']
        # load data split annotations file
        ann_file = find_file_path_in_yaml(self.cfg['ann'], self.root)
        self.anns = json.load(open(ann_file)) # anns = [{}]

        # import pickle
        # n_period = pickle.load(open("tmp/cnt.pk", "rb"))
        # period = len(self.anns)//20
        # if n_period ==19:
        #     self.anns = self.anns[period * n_period: ]
        # elif n_period<19:
        #     self.anns = self.anns[ period*n_period : period*(n_period+1) ]

        # self.rgbd = pickle.load(open("tmp/rgbd.pk", "rb"))
        # n_period = pickle.load(open("tmp/cnt.pk", "rb"))
        # period = len(self.anns) // 200
        # if n_period < 50:
        #     self.anns = self.anns[period * n_period:period*(n_period+1)]

        # few shot
        import random
        random.seed(0)
        random.shuffle(self.anns)
        self.anns = self.anns[:16]
        json.dump(self.anns, open("tmp/anns.json", "w"))

        # load det file
        det_file = find_file_path_in_yaml(self.cfg['det'], self.root)
        self.det_dic = json.load(open(det_file)) # det_dic = {image_id: [box1, box2...]}

        # load image root
        self.image_root = self.cfg['image_root']

        self.transforms = transforms
        # self.demo_color =
        self.colors = [['pink', (255, 205, 225, 160)], ['yellow', (250, 250, 50, 120)]]
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

        imid = self.get_img_id(img_name)
        dets = self.det_dic[imid]
        dets = [d['box'] for d in dets]
        dets = np.asarray(dets)
        dets[:, 2] = dets[:, 0] + dets[:, 2] - 1
        dets[:, 3] = dets[:, 1] + dets[:, 3] - 1
        dets = dets.tolist()

        ret_imgs = []
        ret_targets = []
        ret_color_names = []
        ret_rects = []
        scale = 1.

        # other images
        for i in range( math.ceil( len(dets)/(len(self.colors)-1) ) ):
            img = Image.open(img_path).convert("RGB")
            img_size = img.size

            # find demo det
            iou_list = [self.computeIoU(dets[i], d) for d in dets]
            min_idx = np.argmin(iou_list)
            demo_pos = generate_pos(img_size, dets[min_idx])

            # construct cur dets
            cur_dets = [dets[k] for k in [i, min_idx]]
            target = BoxList(cur_dets, img_size, mode="xyxy")

            # construct colors
            cur_colors = self.colors[:len(cur_dets)]
            cur_color_set = [x[1] for x in cur_colors]
            cur_color_names = [x[0] for x in cur_colors]
            cur_color_names[-1] = cur_color_names[-1] + ";" + demo_pos+";"+str(min_idx)
            ret_color_names.append(cur_color_names)

            assert len(cur_color_set) == len(cur_color_names)
            assert len(cur_color_names) == target.bbox.shape[0]
            assert len(cur_color_names) == len(cur_dets)

            # process img and targets
            # assert target.bbox.shape[0] == len(cur_color_set), "{}  {}   {}".format(anchor_det, target.bbox, cur_color_set)
            self.draw_rectangles(img, target, cur_color_set, imid=ann['id'])
            # add all dets to provide enough context
            target = BoxList(dets, img_size, mode="xyxy")
            img, target = self.transforms(img, target)
            ret_imgs.append(img)
            ret_targets.append(target)
            ret_rects.append(cur_dets)

            new_img_size = img.shape[1:]
            scale = math.sqrt(float(new_img_size[0] * new_img_size[1]) / float(img_size[0] * img_size[1]))
        return ret_imgs, ret_targets, caption, ret_color_names, ret_rects, idx, scale

    def choose_anchor(self, dets):
        return dets.pop(0)

    def draw_rectangles(self, img, target, color_set, imid):
        for i, color in enumerate(color_set):
            box = target.bbox[i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            foreground = Image.new('RGBA', (x2-x1, y2-y1), color=color)
            img.paste(foreground, (x1, y1), foreground)


    def get_img_key(self, idx):
        return self.anns[idx]['id']

    def computeIoU(self, box1, box2):
        # each box is of [x1, y1, x2, y2]
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2], box2[2])
        inter_y2 = min(box1[3], box2[3])

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
        else:
            inter = 0
        union = (box1[2]-box1[0]) * (box1[3]-box1[1]) + (box2[2]-box2[0]) * (box2[3]-box2[1]) - inter
        return float(inter) / union


def generate_pos(img_size, box):
    w, h = img_size
    w_intvl, h_intvl = w/3, h/3
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    n = 0
    for i in range(3):
        for j in range(3):
            if i*w_intvl <= cx <= (i+1)*w_intvl \
                    and j*h_intvl <= cy <= (j+1)*h_intvl:
                n = j*3 + i
    pos_names = ['top left', 'top', 'top right', 'left', 'center', 'right', 'bottom left', 'bottom', 'bottom right']
    return pos_names[n]

