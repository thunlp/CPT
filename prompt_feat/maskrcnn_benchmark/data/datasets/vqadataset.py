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
import copy
import itertools
import pycocotools.mask as mask_util


class VQANormalDataset(object):
    def __init__(self, yaml_file, transforms=None, **kwargs):
        self.cfg = load_from_yaml_file(yaml_file)
        self.root = self.cfg['ann_root']
        ann_file = find_file_path_in_yaml(self.cfg['ann'], self.root)

        self.anns = json.load(open(ann_file))  # anns = [{}]
        print(len(self.anns))
        import pickle
        n_period = pickle.load(open("tmp/cnt.pk", "rb"))
        period = len(self.anns) // 15
        if n_period == 14:
            self.anns = self.anns[period * n_period:]
        elif n_period < 14:
            self.anns = self.anns[period * n_period: period * (n_period + 1)]

        self.transforms = transforms

    def __len__(self):
        return len(self.anns)

    def get_img_info(self, idx):
        return {'width': self.anns[idx]['w'], 'height': self.anns[idx]['h']}

    def __getitem__(self, idx):
        ann = self.anns[idx]
        img_path = ann["img_path"]
        img = Image.open(img_path).convert("RGB")
        img_size = img.size  # w, h
        # target = BoxList([[0., 0, img_size[0], img_size[1]]], image_size=img_size, mode="xyxy")
        target = None
        img, target = self.transforms(img, target)
        new_img_size = img.shape[1:]
        scale = math.sqrt(float(new_img_size[0] * new_img_size[1]) / float(img_size[0] * img_size[1]))
        return img, target, idx, scale

    def get_img_key(self, idx):
        return self.anns[idx]["img_id"]


class VQAColorDataset(object):
    def __init__(self, yaml_file, transforms=None, **kwargs):
        """Constructor.
        Args:
            img_file: Image file with image key and base64 encoded image str.
            label_file: An optional label file with image key and label information.
                A label_file is required for training and optional for testing.
            hw_file: An optional file with image key and image height/width info.
            linelist_file: An optional file with a list of line indexes to load samples.
                It is useful to select a subset of samples or duplicate samples.
        """
        self.cfg = load_from_yaml_file(yaml_file)
        self.root = self.cfg['ann_root']
        ann_file = find_file_path_in_yaml(self.cfg['ann'], self.root)

        self.image_root = self.cfg['image_root']

        self.anns = json.load(open(ann_file))  # anns = [{}]

        det_file = find_file_path_in_yaml(self.cfg['det'], self.root)
        self.det_dic = json.load(open(det_file))  # det_dic = {image_id: [box1, box2...]}

        to_mark_file = find_file_path_in_yaml(self.cfg['to_mark'], self.root)
        self.to_mark = json.load(open(to_mark_file))

        # few shot
        cfg = kwargs.pop("args")
        n_shot = cfg.N_SHOT
        random_seed = cfg.RAND_SEED
        self.n_shot = None
        if n_shot is not None:
            self.n_shot = n_shot
            import random
            assert type(random_seed) is int
            random.seed(random_seed)
            random.shuffle(self.anns)
            self.anns = self.anns[:n_shot]

        # # step by step load
        # cfg = kwargs.pop("args")
        total_step = cfg.TOTAL_STEP
        cur_step = cfg.CUR_STEP
        if total_step is not None and cur_step is not None:
            json.dump([total_step, cur_step], open("tmp/step.json", "w"))
            period = len(self.anns) // total_step
            n_period = cur_step
            if n_period == total_step - 1:
                self.anns = self.anns[period * n_period:]
            elif n_period < total_step - 1:
                self.anns = self.anns[period * n_period: period * (n_period + 1)]

        self.all_colors = [['blue', (0, 10, 255, 127)], ['red', (240, 0, 30, 127)],
                       ['yellow', (255, 255, 25, 127)], ['blue', (0, 10, 255, 127)],
                       ['purple', (155, 50, 210, 127)], ['green', (0, 255, 0, 127)], ]

        self.colors = copy.deepcopy(self.all_colors)
        # self.colors = self.colors[0:1]*100
        self.n_color = 1

        self.anns = [x for x in self.anns if str(x["img_id"]) in self.det_dic]
        # remove uncolored ones TODO
        self.anns = [ann for ann in self.anns if str(ann['qid']) in self.to_mark]

        self.index = 0
        self.transforms = transforms

    def __len__(self):
        return len(self.anns)

    def get_img_info(self, idx):
        return {'width': self.anns[idx]['w'], 'height': self.anns[idx]['h']}

    def __getitem__(self, idx):
        ann = self.anns[idx]
        img_path = ann["img_path"]
        img = Image.open(img_path).convert("RGB")
        img_size = img.size  # w, h

        img_id = ann["img_id"]
        dets = self.det_dic[str(img_id)]
        dets = [d[1] for d in dets]

        # construct to_mark TODO
        to_mark, colors = self.construct_to_mark(ann, img_size)
        if len(to_mark) > 0:
            self.draw_rectangles(img, BoxList(to_mark, img_size, mode="xyxy"), self.colors, save_name=str(ann["qid"]))
        target = BoxList(dets, img_size, mode="xyxy")
        img, target = self.transforms(img, target)

        # convert question to real question
        return img, target, idx, 1.0, colors, None

    def get_img_key(self, idx):
        return self.anns[idx]["qid"]

    def draw_rectangles(self, img, target, color_set, save_name=None):
        for i, box in enumerate(target.bbox):
            color = color_set[i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            foreground = Image.new('RGBA', (max(x2-x1+1, 1), max(y2-y1+1, 1)), color=color[1])
            img.paste(foreground, (x1, y1), foreground)
        # self.index += 1
        # path = "/data_local/zhangao/codes/prompt_feat/tmp/imgs"
        # path = os.path.join(path, str(1))
        # if not os.path.exists(path):
        #     os.mkdir(path)
        # path = os.path.join(path, save_name + ".jpg")
        # img.save(path)


    def construct_to_mark(self, ann, img_size,):
        qid = ann['qid']

        def recover_box(box, img_size):
            w, h = img_size[0]/512, img_size[1]/512
            return [box[0]*w, box[1]*h, box[2]*w, box[3]*h]

        # construct dets
        to_mark = self.to_mark[str(qid)] # [    [[75.0, 359.0, 105.0, 444.0], 4]...    ]
        to_mark = to_mark[: self.n_color]
        dets = [recover_box(x[0], img_size) for x in to_mark]

        # construct positions and color
        colors = [[x[1], color[0]] for x, color in zip(to_mark, self.colors[:self.n_color])]
        return dets, colors



class VQAAugDataset(object):
    def __init__(self, yaml_file, transforms=None, **kwargs):
        """Constructor.
        Args:
            img_file: Image file with image key and base64 encoded image str.
            label_file: An optional label file with image key and label information.
                A label_file is required for training and optional for testing.
            hw_file: An optional file with image key and image height/width info.
            linelist_file: An optional file with a list of line indexes to load samples.
                It is useful to select a subset of samples or duplicate samples.
        """
        self.cfg = load_from_yaml_file(yaml_file)
        self.root = self.cfg['ann_root']
        ann_file = find_file_path_in_yaml(self.cfg['ann'], self.root)

        self.image_root = self.cfg['image_root']

        self.anns = json.load(open(ann_file))  # anns = [{}]

        det_file = find_file_path_in_yaml(self.cfg['det'], self.root)
        self.det_dic = json.load(open(det_file))  # det_dic = {image_id: [box1, box2...]}

        # few shot
        cfg = kwargs.pop("args")
        n_shot = cfg.N_SHOT
        random_seed = cfg.RAND_SEED
        self.n_shot = None
        if n_shot is not None:
            self.n_shot = n_shot
            import random
            assert type(random_seed) is int
            random.seed(random_seed)
            np.random.seed(random_seed)
            random.shuffle(self.anns)
            self.anns = self.anns[:n_shot]

        # # step by step load
        # cfg = kwargs.pop("args")
        total_step = cfg.TOTAL_STEP
        cur_step = cfg.CUR_STEP
        if total_step is not None and cur_step is not None:
            json.dump([total_step, cur_step], open("tmp/step.json", "w"))
            period = len(self.anns) // total_step
            n_period = cur_step
            if n_period == total_step - 1:
                self.anns = self.anns[period * n_period:]
            elif n_period < total_step - 1:
                self.anns = self.anns[period * n_period: period * (n_period + 1)]

        self.all_colors = [['blue', (0, 10, 255, 127)], ['red', (240, 0, 30, 127)],
                       ['yellow', (255, 255, 25, 127)], ['blue', (0, 10, 255, 127)],
                       ['purple', (155, 50, 210, 127)], ['green', (0, 255, 0, 127)], ]

        self.colors = copy.deepcopy(self.all_colors)
        # self.colors = self.colors[0:1]*100
        self.n_color = 1

        self.anns = [x for x in self.anns if str(x["img_id"]) in self.det_dic]

        self.index = 0
        self.transforms = transforms

    def __len__(self):
        return len(self.anns)

    def get_img_info(self, idx):
        return {'width': self.anns[idx]['w'], 'height': self.anns[idx]['h']}

    def __getitem__(self, idx):
        ann = self.anns[idx]
        img_path = ann["img_path"]
        img = Image.open(img_path).convert("RGB")
        img_size = img.size  # w, h

        img_id = ann["img_id"]
        raw_dets = self.det_dic[str(img_id)]
        dets = [d[1] for d in raw_dets]

        # construct to_mark TODO
        to_mark, colors = self.construct_to_mark(raw_dets)
        if len(to_mark) > 0:
            self.draw_rectangles(img, BoxList(to_mark, img_size, mode="xyxy"), self.colors, save_name=str(ann["qid"]))
        target = BoxList(dets, img_size, mode="xyxy")
        img, target = self.transforms(img, target)

        # convert question to real question
        return img, target, idx, 1.0, colors, None

    def get_img_key(self, idx):
        return self.anns[idx]["qid"]

    def draw_rectangles(self, img, target, color_set, save_name=None):
        for i, box in enumerate(target.bbox):
            color = color_set[i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            foreground = Image.new('RGBA', (max(x2-x1+1, 1), max(y2-y1+1, 1)), color=color[1])
            img.paste(foreground, (x1, y1), foreground)
        # self.index += 1
        # path = "/data_local/zhangao/codes/prompt_feat/tmp/imgs"
        # path = os.path.join(path, str(1))
        # if not os.path.exists(path):
        #     os.mkdir(path)
        # path = os.path.join(path, save_name + ".jpg")
        # img.save(path)

    def construct_to_mark(self, dets):
        d = random.choices(dets, k=self.n_color)
        ret_dets = [x[1] for x in d]
        ret_colors = [[x[0], self.colors[i][0]] for i, x in enumerate(d)]
        return ret_dets, ret_colors
