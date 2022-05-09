# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import cv2
import math
import json, pickle
from PIL import Image
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
import os.path as op
import numpy as np
import torch
from maskrcnn_benchmark.structures.tsv_file import TSVFile
from .utils.load_files import load_linelist_file, load_from_yaml_file
from .utils.load_files import find_file_path_in_yaml
from .utils.image_ops import img_from_base64
import glob
import os
from PIL import Image, ImageDraw
import random
import pycocotools.mask as mask_util

class VGDataset(object):
    def __init__(self, yaml_file, transforms=None, **kwargs):
        self.cfg = load_from_yaml_file(yaml_file)
        ann_root = self.cfg["ann_root"]

        ann_path = self.cfg["ann"]
        self.vgdata = pickle.load(open(os.path.join(ann_root, ann_path), "rb"))

        vocab_path = self.cfg["vocab"]
        self.vocab = json.load(open(os.path.join(ann_root, vocab_path)))

        # load image root
        self.image_root = self.cfg["image_root"]

        cfg = kwargs.pop("args")
        n_shot = cfg.N_SHOT
        random_seed = cfg.RAND_SEED
        self.train = n_shot is not None

        if n_shot is not None:
            random.seed(random_seed)
            na_list = random.sample(self.vgdata, n_shot)
            for d in na_list:
                rels = d["relations"]
                pairs = rels[:, :2]
                tmp = np.eye( len(d["boxes"]), dtype=np.int32)
                tmp[pairs[:, 0], pairs[:, 1]] = 1
                na_pairs = (tmp == 0).nonzero()
                na_pairs = np.stack(na_pairs + (np.zeros(na_pairs[0].shape[0]), ), 1)
                rels = np.concatenate([rels, na_pairs], 0).astype(np.int32)
                d["relations"] = rels
            self.vgdata = self.process_vg_train_data(self.vgdata)
        else:
            self.vgdata = self.process_vg_data(self.vgdata)
            random.seed(0)
            random.shuffle(self.vgdata)
            pickle.dump(self.vgdata, open("vg.pk", "wb"))

        # step by step load
        total_step = cfg.TOTAL_STEP
        cur_step = cfg.CUR_STEP
        if total_step is not None and cur_step is not None:
            period = len(self.vgdata)//total_step
            n_period = cur_step
            if n_period == total_step-1:
                self.vgdata = self.vgdata[period * n_period: ]
            elif n_period < total_step-1:
                self.vgdata = self.vgdata[ period*n_period : period*(n_period+1) ]

        # few shot
        if n_shot is not None:
            assert type(random_seed) is int
            # construct rel_dic
            np.random.seed(random_seed)
            rel_dic = {}
            for i, d in enumerate(self.vgdata):
                rel = d['rel']
                if rel not in rel_dic:
                    rel_dic[rel] = []
                rel_dic[rel].append(i)
            new_data = []
            print(len(rel_dic))
            for v in rel_dic.values():
                if len(v) < n_shot:
                    v = v*20
                v = random.sample(v, n_shot)
                new_data.extend([self.vgdata[idx] for idx in v])
            self.vgdata = new_data


        self.transforms = transforms
        self.color_names = ['red', 'blue']
        self.colors = [(240, 0, 30, 127), (0, 10, 255, 127)]
        self.index = 0

    def __len__(self):
        return len(self.vgdata)

    def process_vg_data(self, vgdata):
        toid = lambda name, pair: "_".join([name, str(pair[0]), str(pair[1])])
        idx2pred = lambda x: [self.vocab["idx_to_label"][str(int(ele))] for ele in x]
        new_vgdata = []
        for d in vgdata:
            boxlist = BoxList(torch.from_numpy(d['boxes']), image_size=(d['width'], d['height']), mode="xyxy")
            iou = boxlist_iou(boxlist, boxlist)
            iou[torch.arange(len(iou)), torch.arange(len(iou))] = -1
            idxs = (iou > 0).nonzero().tolist()
            idxs = [t for t in idxs if t[0]<t[1]]
            for t in idxs:
                new_vgdata.append({'width': d['width'], 'height':d['height'], 'img_path':d['img_path'],
                                   'boxes': d['boxes'], 'labels': d['labels'][t], "pair": t,
                                   "pair_labels": idx2pred(d['labels'][t]),
                                   "id": toid(d['img_path'], t)})
        return new_vgdata

    def process_vg_train_data(self, vgdata):
        toid = lambda name, pair: "_".join([name, str(pair[0]), str(pair[1])])
        idx2label = lambda x: [self.vocab["idx_to_label"][str(int(ele))] for ele in x]
        self.vocab["idx_to_predicate"]["0"] = "irrelevant"
        idx2pred = lambda x: self.vocab["idx_to_predicate"][str(x)]
        new_vgdata = []
        for d in vgdata:
            boxlist = BoxList(torch.from_numpy(d['boxes']), image_size=(d['width'], d['height']), mode="xyxy")
            idxs = d["relations"].tolist()
            for t in idxs:
                r = t[-1]
                t = t[:2]
                new_vgdata.append({'width': d['width'], 'height':d['height'], 'img_path':d['img_path'],
                                   'boxes': d['boxes'], 'labels': d['labels'][t], "pair": t,
                                   "pair_labels": idx2label(d['labels'][t]), "rel": idx2pred(r),
                                   "id": toid(d['img_path'], t)})
        return new_vgdata

    def __getitem__(self, idx):
        ann = self.vgdata[idx]

        img_name = ann['img_path']
        img_path = os.path.join(self.image_root, img_name)

        dets = ann["boxes"]
        img = Image.open(img_path).convert("RGB")
        img_size = img.size
        colors = self.colors

        self.draw_rectangles(img, BoxList(dets[ann['pair']], img_size, mode="xyxy"), colors)
        target = BoxList(dets, img_size, mode="xyxy")
        img, target = self.transforms(img, target)
        rel = None
        if "rel" in ann:
            rel = ann["rel"]
        return img, target, self.color_names, ann["pair_labels"], idx, rel

    def draw_rectangles(self, img, target, color_set):
        for i, color in enumerate(color_set):
            box = target.bbox[i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            foreground = Image.new('RGBA', (x2-x1, y2-y1), color=color)
            img.paste(foreground, (x1, y1), foreground)

    def get_img_key(self, idx):
        return self.vgdata[idx]['id']

    def get_img_info(self, idx):
        data = self.vgdata[idx]
        return {'width': data['width'], 'height': data['height']}


class VGNormalDataset(object):
    def __init__(self, yaml_file, transforms=None, **kwargs):
        cfg = kwargs.pop("args")
        n_shot = cfg.N_SHOT
        random_seed = cfg.RAND_SEED
        self.train = n_shot is not None

        self.vocab = json.load(open("data/vg/vg.json"))
        if n_shot is not None:
            random.seed(random_seed)
            self.vgdata = pickle.load(open("data/vg/vg_train.pk", "rb"))
            na_list = random.sample(self.vgdata, n_shot)
            for d in na_list:
                rels = d["relations"]
                pairs = rels[:, :2]
                tmp = np.eye( len(d["boxes"]), dtype=np.int32)
                tmp[pairs[:, 0], pairs[:, 1]] = 1
                na_pairs = (tmp == 0).nonzero()
                na_pairs = np.stack(na_pairs + (np.zeros(na_pairs[0].shape[0]), ), 1)
                rels = np.concatenate([rels, na_pairs], 0).astype(np.int32)
                d["relations"] = rels
            self.vgdata = self.process_vg_train_data(self.vgdata)
        else:
            self.vgdata = pickle.load(open("data/vg/vg_val.pk", "rb"))
            self.vgdata = self.process_vg_data(self.vgdata)

        # step by step load
        total_step = cfg.TOTAL_STEP
        cur_step = cfg.CUR_STEP
        if total_step is not None and cur_step is not None:
            period = len(self.vgdata)//total_step
            n_period = cur_step
            if n_period == total_step-1:
                self.vgdata = self.vgdata[period * n_period: ]
            elif n_period < total_step-1:
                self.vgdata = self.vgdata[ period*n_period : period*(n_period+1) ]

        # few shot
        if n_shot is not None:
            assert type(random_seed) is int
            # construct rel_dic
            np.random.seed(random_seed)
            rel_dic = {}
            for i, d in enumerate(self.vgdata):
                rel = d['rel']
                if rel not in rel_dic:
                    rel_dic[rel] = []
                rel_dic[rel].append(i)
            new_data = []
            for v in rel_dic.values():
                if n_shot > len(v):
                    v = v*20
                v = random.sample(v, n_shot)
                new_data.extend([self.vgdata[idx] for idx in v])
            self.vgdata = new_data

        # load image root
        self.image_root = "../Long-Tail-VRD.pytorch"

        self.transforms = transforms
        self.color_names = ['red', 'blue']
        self.colors = [(240, 0, 30, 127), (0, 10, 255, 127)]
        self.index = 0

    def __len__(self):
        return len(self.vgdata)

    def process_vg_data(self, vgdata):
        for d in vgdata:
            boxlist = BoxList(torch.from_numpy(d['boxes']), image_size=(d['width'], d['height']), mode="xyxy")
            iou = boxlist_iou(boxlist, boxlist)
            iou[torch.arange(len(iou)), torch.arange(len(iou))] = -1
            idxs = (iou > 0).nonzero().tolist()
            d["pairs"] = idxs
        return vgdata

    def process_vg_train_data(self, vgdata):
        toid = lambda name, pair: "_".join([name, str(pair[0]), str(pair[1])])
        idx2label = lambda x: [self.vocab["idx_to_label"][str(int(ele))] for ele in x]
        self.vocab["idx_to_predicate"]["0"] = "irrelevant"
        idx2pred = lambda x: self.vocab["idx_to_predicate"][str(x)]
        new_vgdata = []
        for d in vgdata:
            boxlist = BoxList(torch.from_numpy(d['boxes']), image_size=(d['width'], d['height']), mode="xyxy")
            idxs = d["relations"].tolist()
            for t in idxs:
                r = t[-1]
                t = t[:2]
                new_vgdata.append({'width': d['width'], 'height':d['height'], 'img_path':d['img_path'],
                                   'boxes': d['boxes'], 'labels': d['labels'][t], "pairs": [t],
                                   "pair_labels": idx2label(d['labels'][t]), "rel": r,
                                   "id": toid(d['img_path'], t)})
        return new_vgdata

    def __getitem__(self, idx):
        ann = self.vgdata[idx]

        img_name = ann['img_path']
        img_path = os.path.join(self.image_root, img_name)

        dets = ann["boxes"]
        img = Image.open(img_path).convert("RGB")
        img_size = img.size

        target = BoxList(dets, img_size, mode="xyxy")
        img, target = self.transforms(img, target)
        rel = None
        pairs = None
        if "rel" in ann:
            rel = ann["rel"]
        if "pairs" in ann:
            pairs = ann["pairs"]
        return img, target, None, [rel], idx, pairs

    def get_img_key(self, idx):
        return self.vgdata[idx]['img_path']

    def get_img_info(self, idx):
        data = self.vgdata[idx]
        return {'width': data['width'], 'height': data['height']}
