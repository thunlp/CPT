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


class VCRNormalDataset(object):
    def __init__(self, yaml_file, transforms=None, **kwargs):
        self.cfg = load_from_yaml_file(yaml_file)
        self.root = self.cfg['ann_root']
        ann_file = find_file_path_in_yaml(self.cfg['ann'], self.root)

        self.image_root = self.cfg['image_root']

        self.anns = json.load(open(ann_file))  # anns = [{}]

        import pickle
        n_period = pickle.load(open("tmp/cnt.pk", "rb"))
        period = len(self.anns) // 10
        if n_period == 9:
            self.anns = self.anns[period * n_period:]
        elif n_period < 9:
            self.anns = self.anns[period * n_period: period * (n_period + 1)]

        self.transforms = transforms

    def __len__(self):
        return len(self.anns)

    def get_img_info(self, idx):
        return {'width': self.anns[idx]['w'], 'height': self.anns[idx]['h']}

    def __getitem__(self, idx):
        ann = self.anns[idx]
        img_path = os.path.join(self.image_root, ann["img_path"])
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


class VCRColorDataset(object):
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

        to_mark_file = find_file_path_in_yaml(self.cfg['to_mark'], self.root)
        self.to_mark_dic = json.load(open(to_mark_file))

        det_file = find_file_path_in_yaml(self.cfg['det'], self.root)
        self.det_dic = json.load(open(det_file))  # det_dic = {image_id: [box1, box2...]}

        self.seg_dic = True
        # if "seg" in self.cfg:
        #     seg_file = find_file_path_in_yaml(self.cfg['seg'], self.root)
        #     self.seg_dic = json.load(open(seg_file))

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
        self.n_color = int(cfg.COLOR_D)

        self.anns = [x  for x in self.anns if x["img_id"] in self.det_dic]
        self.index = 0
        self.transforms = transforms

    def __len__(self):
        return len(self.anns)

    def get_img_info(self, idx):
        return {'width': self.anns[idx]['w'], 'height': self.anns[idx]['h']}

    def __getitem__(self, idx):
        ann = self.anns[idx]
        img_path = os.path.join(self.image_root, ann["img_path"])
        img = Image.open(img_path).convert("RGB")
        img_size = img.size  # w, h

        img_id = ann["img_id"]
        dets = self.det_dic[img_id]
        dets = [d[1] for d in dets]

        # construct to_mark
        to_mark_boxes_and_names = self.to_mark_dic[img_id]
        to_mark_ids, to_mark, colors, txt_colors, txt_names, _ = \
            self.construct_to_mark_and_color(ann, to_mark_boxes_and_names["boxes"], to_mark_boxes_and_names["names"])
        # print(ann["annot_id"], to_mark_ids, colors, txt_colors, txt_names, _ )
        # dets = to_mark + dets

        if len(to_mark) > 0:
            if self.seg_dic:
                masks = json.load(open(os.path.join(self.image_root, ann["img_path"].replace(".jpg", ".json"))))
                masks = [masks["segms"][i] for i in to_mark_ids]
                # masks = [self.seg_dic[img_id]["boxes"][i] for i in to_mark_ids]
                self.draw_segmasks(img, masks, colors, save_name=str(ann["annot_id"]))
            else:
                self.draw_rectangles(img, BoxList(to_mark, img_size, mode="xyxy"), colors, save_name=str(ann["annot_id"]))
        target = BoxList(dets, img_size, mode="xyxy")
        img, target = self.transforms(img, target)

        # convert question to real question
        return img, target, idx, 1.0, txt_colors, txt_names

    def get_img_key(self, idx):
        return self.anns[idx]["annot_id"]

    def construct_to_mark_and_color(self, ann, to_mark_list, to_mark_names):
        def _get_eles(sentence, lst2str):
            ret = [lst2str(x) for x in sentence if type(x) is list]
            ret = list(set(ret))
            return ret
        lst2str = lambda x: "_".join([str(y) for y in sorted(x)])
        str2list = lambda x: [int(y) for y in x.split("_")]
        q, anws, rations = ann["question"], ann["answers"], ann["rationales"]
        # txt to mark elements
        dic = {}
        all_eles = _get_eles(q, lst2str) + [y for x in anws for y in _get_eles(x, lst2str)] \
                   + [y for x in rations for y in _get_eles(x, lst2str)]
        to_mark_dets = []
        to_mark_cls = []
        to_mark = []
        for ele in all_eles:
            if ele not in dic:
                dic[ele] = 0
                to_mark.append(str2list(ele))
                to_mark_dets.append([to_mark_list[i][:4] for i in str2list(ele)])
                to_mark_cls.append([to_mark_names[i] for i in str2list(ele)])

        # vis to mark elements
        dic = {}
        vis_eles = _get_eles(q, lst2str) + [y for x in anws for y in _get_eles(x, lst2str)]
                   # + [y for x in rations for y in _get_eles(x, lst2str)]
        vis_to_mark_dets = []
        vis_to_mark = []
        for ele in vis_eles:
            if ele not in dic:
                dic[ele] = 0
                vis_to_mark.append(str2list(ele))
                vis_to_mark_dets.append([to_mark_list[i][:4] for i in str2list(ele)])

        #
        ret_to_mark_ids = []
        ret_to_mark_dets = []
        ret_vis_colors = []
        ret_txt_colors = {}
        ret_txt_names = {}
        ele_color_dic = {}
        color_cnt = 0
        # to_grd = self.to_grd.get(ann["annot_id"], None)
        for i, (m_list, dets) in enumerate(zip(vis_to_mark, vis_to_mark_dets)):

            # if to_grd is not None and m_list not in to_grd:
            #     continue
            # continue
            if color_cnt >= self.n_color:
                continue

            # for dataset to paint colors
            paint_or_not = False
            for m, d in zip(m_list, dets):
                if d not in ret_to_mark_dets:
                    ret_to_mark_ids.append(m)
                    ret_vis_colors.append(self.colors[color_cnt])
                    ret_to_mark_dets.append(d)
                    ele_color_dic[m] = self.colors[color_cnt]
                    paint_or_not = True
            if paint_or_not:
                color_cnt += 1

        # txt to mark names and colors
        for m_list, m_clses in zip(to_mark, to_mark_cls):
            # for to_mark_cls
            cls_set = set(m_clses)
            # [person, person] or [dog, dog]
            if len(cls_set) == 1:
                if m_clses[0] == "person":
                    ret_txt_names[lst2str(m_list)] = "person" if len(m_clses) == 1 else "people"
                else:
                    ret_txt_names[lst2str(m_list)] = m_clses[0] if len(m_clses) == 1 else m_clses[0] + "s"
            # [person, dog]
            else:
                ret_txt_names[lst2str(m_list)] = "objects"

            # for txt_colors
            cur_m_colors = set([ele_color_dic.get(m, ["none"])[0] for m in m_list])
            if len(cur_m_colors) == 1 and list(cur_m_colors)[0] != "none":
                ret_txt_colors[lst2str(m_list)] = list(cur_m_colors)[0]

        return ret_to_mark_ids, ret_to_mark_dets, ret_vis_colors, ret_txt_colors, ret_txt_names, color_cnt


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

    def draw_segmasks(self, img, segms, color_set, save_name=None):
        for i, segm in enumerate(segms):
            color = color_set[i]
            overlay = Image.new('RGBA', img.size, tuple(color[1][:3]) + (0,))
            draw = ImageDraw.Draw(overlay)
            for segm_part in segm:
                if len(segm_part) < 2:
                    segm_part += tuple([segm_part[0]])

                segm_part = tuple(tuple(x) for x in segm_part)
                draw.polygon(segm_part, fill=color[1])
            # img = Image.alpha_composite(img, overlay).convert("RGB")
            img.paste(overlay, (0, 0), overlay)


    def _get_to_annot_color(self, idx):
        ann = self.anns[idx]
        img_id = ann["img_id"]
        to_mark_boxes_and_names = self.to_mark_dic[img_id]
        to_mark_ids, to_mark, colors, txt_colors, txt_names, color_cnt = \
            self.construct_to_mark_and_color(ann, to_mark_boxes_and_names["boxes"], to_mark_boxes_and_names["names"])
        return color_cnt