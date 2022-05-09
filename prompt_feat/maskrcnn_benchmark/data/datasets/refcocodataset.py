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
import pycocotools.mask as mask_util


class ImgDataset(object):
    def __init__(self, data_dir, transforms=None, **kwargs):
        """Constructor.
        Args:
            img_file: Image file with image key and base64 encoded image str.
            label_file: An optional label file with image key and label information.
                A label_file is required for training and optional for testing.
            hw_file: An optional file with image key and image height/width info.
            linelist_file: An optional file with a list of line indexes to load samples.
                It is useful to select a subset of samples or duplicate samples.
        """
        self.img_files = glob.glob(os.path.join(data_dir, "*.jpg"))
        self.img_files = sorted(self.img_files)

        import pickle
        n_period = pickle.load(open("tmp/cnt.pk", "rb"))
        period = len(self.img_files) // 15
        if n_period == 14:
            self.img_files = self.img_files[period * n_period:]
        elif n_period < 14:
            self.img_files = self.img_files[period * n_period: period * (n_period + 1)]

        self.img_info = json.load(open(os.path.join(data_dir, "../img_info.json")))
        self.transforms = transforms

    def __len__(self):
        return len(self.img_files)

    def get_img_info(self, idx):
        return self.img_info[os.path.basename(self.img_files[idx])]

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx]).convert("RGB")
        img_size = img.size # w, h
        # target = BoxList([[0., 0, img_size[0], img_size[1]]], image_size=img_size, mode="xyxy")
        target = None
        img, target = self.transforms(img, target)
        new_img_size = img.shape[1:]
        scale = math.sqrt(float(new_img_size[0]*new_img_size[1])/float(img_size[0]*img_size[1]))
        return img, target, idx, scale

    def get_img_key(self, idx):
        return os.path.basename(self.img_files[idx]).replace(".jpg", "")



class NormalFinetuneDataset(object):
    def __init__(self, yaml_file, transforms=None, **kwargs):
        self.cfg = load_from_yaml_file(yaml_file)
        self.root = self.cfg['ann_root']
        # load data split annotations file
        ann_file = find_file_path_in_yaml(self.cfg['ann'], self.root)
        self.anns = json.load(open(ann_file))  # anns = [{}]

        # step by step load
        cfg = kwargs.pop("args")
        total_step = cfg.TOTAL_STEP
        cur_step = cfg.CUR_STEP
        if total_step is not None and cur_step is not None:
            period = len(self.anns) // total_step
            n_period = cur_step
            if n_period == total_step - 1:
                self.anns = self.anns[period * n_period:]
            elif n_period < total_step - 1:
                self.anns = self.anns[period * n_period: period * (n_period + 1)]

        # few shot
        n_shot = cfg.N_SHOT
        random_seed = cfg.RAND_SEED
        if n_shot is not None:
            # np.random.seed(random_seed)
            # self.anns = np.random.choice(self.anns, n_shot)
            import random
            assert type(random_seed) is int
            random.seed(random_seed)
            random.shuffle(self.anns)
            self.anns = self.anns[:n_shot]

        # load det file
        det_file = find_file_path_in_yaml(self.cfg['det'], self.root)
        self.det_dic = json.load(open(det_file))  # det_dic = {image_id: [box1, box2...]}

        # load image root
        self.image_root = self.cfg['image_root']

        self.transforms = transforms

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

        # get max iou bounding box
        dets = np.asarray(dets)
        # convert from (x1, y1, w, h) to (x1, y1, x2, y2)
        dets[:, 2] = dets[:, 0] + dets[:, 2] - 1
        dets[:, 3] = dets[:, 1] + dets[:, 3] - 1
        # dets = dets.tolist()
        mask = (dets[:, 0] - dets[:, 2]) >= 0
        dets[:, 2][mask] += 1
        mask = (dets[:, 1] - dets[:, 3]) >= 0
        dets[:, 3][mask] += 1
        assert (dets[:, 0]<dets[:,2]).all() and (dets[:, 1]<dets[:,3]).all()

        img = Image.open(img_path).convert("RGB")
        img_size = img.size
        target = BoxList(dets, img_size, mode="xyxy")
        img, target = self.transforms(img, target)
        new_img_size = img.shape[1:]
        scale = math.sqrt(float(new_img_size[0] * new_img_size[1]) / float(img_size[0] * img_size[1]))
        return [img], [target], caption, [None], [None], idx, scale

    def get_img_key(self, idx):
        return self.anns[idx]['id']

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


class RefCoCoDataset(object):
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
        # load data split annotations file
        ann_file = find_file_path_in_yaml(self.cfg['ann'], self.root)
        self.anns = json.load(open(ann_file)) # anns = [{}]
        self.is_train = "train" in self.cfg['ann']

        # step by step load
        cfg = kwargs.pop("args")
        total_step = cfg.TOTAL_STEP
        cur_step = cfg.CUR_STEP
        if total_step is not None and cur_step is not None:
            period = len(self.anns)//total_step
            n_period = cur_step
            if n_period == total_step-1:
                self.anns = self.anns[period * n_period: ]
            elif n_period < total_step-1:
                self.anns = self.anns[ period*n_period : period*(n_period+1) ]

        # few shot
        n_shot = cfg.N_SHOT
        random_seed = cfg.RAND_SEED
        if n_shot is not None:
            import random
            assert type(random_seed) is int
            random.seed(random_seed)
            random.shuffle(self.anns)
            self.anns = self.anns[:n_shot]

        # load det file
        det_file = find_file_path_in_yaml(self.cfg['det'], self.root)
        self.det_dic = json.load(open(det_file)) # det_dic = {image_id: [box1, box2...]}

        # load image root
        self.image_root = self.cfg['image_root']

        self.transforms = transforms
        self.colors = [['red', (240, 0, 30, 127)]]
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
        rawdets = self.det_dic[imid]
        dets = [d['box'] for d in rawdets]
        masks = None
        if "rle" in rawdets[0]:
            masks = [d["rle"] for d in rawdets]
        else:
            masks = [None for d in rawdets]
        # if self.is_train:
        #     gt = ann['bbox']
        #     dets.append(gt)
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
        for i in range( math.ceil( len(dets)/len(self.colors) ) ):
            img = Image.open(img_path).convert("RGB")
            img_size = img.size
            # construct cur dets
            cur_dets = dets[i*len(self.colors): (i+1)*len(self.colors)]
            target = BoxList(cur_dets, img_size, mode="xyxy")

            # construct colors
            cur_colors = self.colors[:len(cur_dets)]
            cur_color_set = [x[1] for x in cur_colors]
            cur_color_names = [x[0] for x in cur_colors]
            ret_color_names.append(cur_color_names)

            assert len(cur_color_set) == len(cur_color_names)
            assert len(cur_color_names) == target.bbox.shape[0]
            assert len(cur_color_names) == len(cur_dets)

            # process img and targets
            self.draw_rectangles(img, target, cur_color_set, mask=masks[i], imid=ann['id'])
            # add all dets to provide enough context
            target = BoxList(dets, img_size, mode="xyxy")
            img, target = self.transforms(img, target)
            ret_imgs.append(img)
            ret_targets.append(target)
            ret_rects.append(cur_dets)

            new_img_size = img.shape[1:]
            scale = math.sqrt(float(new_img_size[0] * new_img_size[1]) / float(img_size[0] * img_size[1]))
        return ret_imgs, ret_targets, caption, ret_color_names, ret_rects, idx, scale

    def draw_rectangles(self, img, target, color_set, imid=None, mask=None):
        for i, color in enumerate(color_set):
            if mask:
                m = mask_util.decode(mask)
                m = m.reshape(m.shape[0], m.shape[1], 1).repeat(4, axis=-1)
                m = m*np.array(color)
                foreground = Image.fromarray(m.astype(np.uint8)).convert("RGBA")
                x1, y1 = 0, 0
            else:
                box = target.bbox[i]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                foreground = Image.new('RGBA', (x2-x1, y2-y1), color=color)
            img.paste(foreground, (x1, y1), foreground)

    def get_img_key(self, idx):
        return self.anns[idx]['id']


class ValDataset(object):
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
        # load data split annotations file
        ann_file = find_file_path_in_yaml(self.cfg['ann'], self.root)
        self.anns = json.load(open(ann_file)) # anns = [{}]
        self.is_train = "train" in self.cfg['ann']

        # step by step load
        cfg = kwargs.pop("args")
        total_step = cfg.TOTAL_STEP
        cur_step = cfg.CUR_STEP
        if total_step is not None and cur_step is not None:
            period = len(self.anns)//total_step
            n_period = cur_step
            if n_period == total_step-1:
                self.anns = self.anns[period * n_period: ]
            elif n_period < total_step-1:
                self.anns = self.anns[ period*n_period : period*(n_period+1) ]

        # few shot
        n_shot = cfg.N_SHOT
        random_seed = cfg.RAND_SEED
        if n_shot is not None:
            import random
            assert type(random_seed) is int
            random.seed(random_seed)
            random.shuffle(self.anns)
            self.anns = self.anns[:n_shot]

        # load det file
        det_file = find_file_path_in_yaml(self.cfg['det'], self.root)
        self.det_dic = json.load(open(det_file)) # det_dic = {image_id: [box1, box2...]}

        # load image root
        self.image_root = self.cfg['image_root']

        self.transforms = transforms
        color_name = cfg.COLOR_NAME
        color_val = cfg.COLOR_RGB
        color_d = cfg.COLOR_D
        color_val = tuple([int(x) for x in color_val.split("-")]) + (color_d, )
        self.colors = [[color_name, color_val]]
        self.index = 0

        json.dump([self.colors, n_shot, random_seed], open("tmp/tmp.json", "w"))

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
        rawdets = self.det_dic[imid]
        dets = [d['box'] for d in rawdets]
        masks = None
        if "rle" in rawdets[0]:
            masks = [d["rle"] for d in rawdets]
        else:
            masks = [None for d in rawdets]
        # if self.is_train:
        #     gt = ann['bbox']
        #     dets.append(gt)
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
        for i in range( math.ceil( len(dets)/len(self.colors) ) ):
            img = Image.open(img_path).convert("RGB")
            img_size = img.size
            # construct cur dets
            cur_dets = dets[i*len(self.colors): (i+1)*len(self.colors)]
            target = BoxList(cur_dets, img_size, mode="xyxy")

            # construct colors
            cur_colors = self.colors[:len(cur_dets)]
            cur_color_set = [x[1] for x in cur_colors]
            cur_color_names = [x[0] for x in cur_colors]
            ret_color_names.append(cur_color_names)

            assert len(cur_color_set) == len(cur_color_names)
            assert len(cur_color_names) == target.bbox.shape[0]
            assert len(cur_color_names) == len(cur_dets)

            # process img and targets
            self.draw_rectangles(img, target, cur_color_set, mask=masks[i], imid=ann['id'])
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

    def draw_rectangles(self, img, target, color_set, imid=None, mask=None):
        for i, color in enumerate(color_set):
            if mask:
                m = mask_util.decode(mask)
                m = m.reshape(m.shape[0], m.shape[1], 1).repeat(4, axis=-1)
                m = m * np.array(color)
                foreground = Image.fromarray(m.astype(np.uint8)).convert("RGBA")
                x1, y1 = 0, 0
            else:
                box = target.bbox[i]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                foreground = Image.new('RGBA', (x2 - x1, y2 - y1), color=color)
            img.paste(foreground, (x1, y1), foreground)
        # self.index += 1
        # path = "/data_local/zhangao/codes/prompt_feat/tmp/imgs"
        # path = os.path.join(path, str(imid))
        # if not os.path.exists(path):
        #     os.mkdir(path)
        # path = os.path.join(path, str(self.index) + ".jpg")
        # img.save(path)

    def get_img_key(self, idx):
        return self.anns[idx]['id']
