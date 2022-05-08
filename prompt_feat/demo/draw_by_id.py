import json
import sys
from PIL import Image, ImageDraw
import random
import pycocotools.mask as mask_util
import numpy as np
import os

mark = "+" if "+" in sys.argv[1] else "g"
thres = {"+": 25112.9428, "g": 20149.3232}
refcoco_dic = json.load(open("tmp/refcoco{}.json".format(mark)))
refcoco_dic = {k: 0 for k in refcoco_dic}
data = json.load(open("data/refcoco/split/finetune_refcoco{}_val.json".format(mark)))
images = {}
for x in data:
    if str(x['id']) not in refcoco_dic:
        continue
    fname = x['file_name']
    if fname not in images:
        images[fname] = []
    images[fname].append(x['bbox'])
det_dic = json.load(open("data/refcoco/detections/refcoco{}/segs.json".format(mark)))
image_root = "/data_local/zhangao/data/coco/train2014"
save_path = "tmp/small"


def get_img_id(name):
    name = name.replace(".jpg", "").split("_")[-1]
    return str(int(name))


def calc_area(det):
    return det[-1]*det[-2]


def draw(img_name, det_dic, gt_boxes):
    imid = get_img_id(img_name)
    rawdets = det_dic[imid]
    dets = [d['box'] for d in rawdets]
    masks = None
    if "rle" in rawdets[0]:
        masks = [d["rle"] for d in rawdets]
    else:
        masks = [None for d in rawdets]
    masks = [(m, has_gt(dets[i], gt_boxes)) for i, m in enumerate(masks)]
    masks = [m for m in masks if m[1] is not None]
    # if len(masks) > 1:
    #     masks = np.random.choice(masks, 1)
    for i, m in enumerate(masks):
        img_path = os.path.join(image_root, img_name)
        img = Image.open(img_path).convert("RGB")
        img = draw_seg(img, m[0], m[1])
        save_dir = os.path.join(save_path, str(imid))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        img.save(os.path.join(save_path, str(imid), str(i)+".jpg"))

def has_gt(det, gts):
    for x in gts:
        if computeIoU(det, x) > 0.5:
            return x
    return None
    # return np.any([computeIoU(det, x)>0.5 for x in gts])

def computeIoU(box1, box2):
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


def draw_seg(img, mask, box=None, color=(240, 0, 30, 127)):
    m = mask_util.decode(mask)
    m = m.reshape(m.shape[0], m.shape[1], 1).repeat(4, axis=-1)
    m = m * np.array(color)
    foreground = Image.fromarray(m.astype(np.uint8)).convert("RGBA")
    x1, y1 = 0, 0
    img.paste(foreground, (x1, y1), foreground)
    if box:
        x1, y1, x2, y2 = box[0], box[1], box[0]+box[2]-1, box[1]+box[3]-1
        draw = ImageDraw.Draw(img)
        draw.rectangle(((x1, y1), (x2, y2)), outline="green", width=2)
    return img


from tqdm import tqdm
for img_name, gts in tqdm(images.items()):
    draw(img_name, det_dic, gts)