import json

anns = json.load(open("/data_local/zhangao/codes/prompt_feat/data/refcoco/split/finetune_refcoco+_testA.json"))
dets = json.load(open("/data_local/zhangao/codes/prompt_feat/data/refcoco/detections/refcoco+/dets.json"))


def computeIoU(box1, box2):
    # each box is of [x1, y1, w, h]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
    inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = box1[2]*box1[3] + box2[2]*box2[3] - inter
    return float(inter)/union


def get_img_id(name):
    name = name.replace(".jpg", "").split("_")[-1]
    return str(int(name))


correct = 0
for ann in anns:
    imid = get_img_id(ann['file_name'])
    gt = ann['bbox']
    # assert gt[0] < gt[2] and gt[1] < gt[3]
    max_iou = 0
    for d in dets[imid]:
        # assert d['box'][0] < d['box'][2] and d['box'][1] < d['box'][3]
        max_iou = max(max_iou, computeIoU(d['box'], gt))
    if len(dets[imid]) == 1:
        correct += (max_iou>0.5)
print(correct, len(anns), correct/len(anns))