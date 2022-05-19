import torch
import json
import pickle
from collections import Counter
import numpy as np
import sys, os
import glob

val_gts = pickle.load(open("../data/vg/vg_val.pk", "rb"))
test_gts = pickle.load(open("../data/vg/vg_test.pk", "rb"))
# test_gts = pickle.load(open("../prompt_feat/data/vg/vg_test.pk", "rb"))
imkey2pair = lambda n: [int(x) for x in n.split("_")[-2:]]


def eval(gts, preds):
    new_preds = {}
    for imkey, prd in preds.items():
        img_name = imkey.split(".jpg")[0]+".jpg"
        pair = imkey2pair(imkey)
        if img_name not in new_preds:
            new_preds[img_name] = []
        new_preds[img_name].append({"pair": pair, "pred": prd})

    for g in gts:
        if g["img_path"] not in new_preds:
            new_preds[g["img_path"]] = [{"pair": [0, 0], "pred": torch.zeros(51, dtype=torch.float)}]

    gts = [g for g in gts if g['img_path'] in new_preds]
    assert len(new_preds) == len(gts), "{}, {}".format(len(new_preds), len(gts))
    preds = [new_preds[k["img_path"]] for k in gts]

    recall = {20:[], 50: [], 100: []}
    mrecall = {20: [[] for i in range(51)], 50: [[] for i in range(51)], 100: [[] for i in range(51)]}

    for p_list, gt in zip(preds, gts):
        pairs = [p['pair'] for p in p_list]
        prds = [p['pred'] for p in p_list]
        pairs = torch.tensor(pairs)
        prds = torch.stack(prds, 0).softmax(1)
        # avoid difference with different torch version
        if (prds[:, 48] == prds[:, 49]).all():
            prds[:, 49] += 1e-5
        rels = prds[:, 1:].argmax(1) + 1
        scores = prds[torch.arange(len(prds)), rels]
        idxs = scores.argsort(descending=True)
        rels = rels[idxs]
        pairs = pairs[idxs]
        rels = torch.cat([pairs, rels[:, None]], -1)

        gt_rels = torch.from_numpy(gt["relations"])
        # calculate recall
        for mode in recall:
            pred_rels = rels[:mode]
            rcl = (gt_rels[:, :, None] == pred_rels.T[None, :, :]).all(1).any(1)
            recall[mode].append(sum(rcl)/float(len(gt_rels)))

            tmp_cnt = Counter(gt_rels[:, 2].tolist())
            tmp_m_recall = {}
            assert len(gt_rels) == len(rcl)
            for r, c in zip(gt_rels[:, 2].tolist(), rcl):
                tmp_m_recall[r] = tmp_m_recall.get(r, 0) + int(c)
            for r in tmp_m_recall:
                mrecall[mode][r].append(tmp_m_recall[r]/tmp_cnt[r])

    recall = {k: np.mean(v) for k, v in recall.items()}
    mrecall = {k: np.mean( [ np.mean(v) if len(v)>0 else 0 for v in v_list[1:] ] ) for k, v_list in mrecall.items()}
    # print()
    # print("R@20: {:.2f}\tR@50: {:.2f}\tR@100: {:.2f}".format(recall[20], recall[50], recall[100]))
    # print("mR@20: {:.2f}\tmR@50: {:.2f}\tmR@100: {:.2f}".format(mrecall[20], mrecall[50], mrecall[100]))
    return np.array([recall[20], recall[50], recall[100], mrecall[20], mrecall[50], mrecall[100]])*100


def print_rst(recall, std=None):
    recall = recall.tolist()
    if std is not None:
        std = std.tolist()
        print("R@20: {:.2f}/{:.2f}\tR@50: {:.2f}/{:.2f}\tR@100: {:.2f}/{:.2f}"
              .format(recall[0], std[0], recall[1], std[1], recall[2], std[2]))
        print("mR@20: {:.2f}/{:.2f}\tmR@50: {:.2f}/{:.2f}\tmR@100: {:.2f}/{:.2f}"
              .format(recall[3], std[3], recall[4], std[4], recall[5], std[5]))
    else:
        print("R@20: {:.2f}\tR@50: {:.2f}\tR@100: {:.2f}".format(recall[0], recall[1], recall[2]))
        print("mR@20: {:.2f}\tmR@50: {:.2f}\tmR@100: {:.2f}".format(recall[3], recall[4], recall[5]))


root = sys.argv[1]

# zsl evaluation
zsl_path = os.path.join(root, "fsl/0/0", "*.pt")
zsl_path = glob.glob(zsl_path)
print("begin zsl evaluation")
if len(zsl_path) == 0:
    print("nothing to evaluate")
for p in zsl_path:
    if "val" in p:
        rst = eval(val_gts, torch.load(p))
        print("val:")
    elif "test" in p:
        rst = eval(test_gts, torch.load(p))
        print("test")
    print_rst(rst)
print("-"*50)

# fsl evaluation
print("begin fsl evaluation")
fsl_path = os.path.join(root, "fsl", "*") # results/vg/cpt/fsl/*
fsl_shot_dir_list = glob.glob(fsl_path) # results/vg/cpt/fsl/[1, 2, 4, 8, 16]
for fsl_shot_dir in fsl_shot_dir_list:
    per_seed_path_list = glob.glob(os.path.join(fsl_shot_dir, "*")) # results/vg/cpt/fsl/16/[0,1,2,3,4]
    val_rsts, test_rsts = [], []
    print("shot {}".format(os.path.basename(fsl_shot_dir)))
    for per_seed_path in per_seed_path_list:
        val_path = os.path.join(per_seed_path, "val.pt")
        if os.path.exists(val_path):
            val_pred = torch.load(val_path)
            val_rsts.append(eval(val_gts, val_pred))
            # print(val_rsts[-1])

        test_path = os.path.join(per_seed_path, "test.pt")
        if os.path.exists(test_path):
            test_pred = torch.load(test_path)
            test_rsts.append(eval(test_gts, test_pred))

    # calculate mean and std
    if len(val_rsts)>0:
        means, stds = np.mean(val_rsts, 0), np.std(val_rsts, 0)
        print("{} seeds".format(len(val_rsts)))
        print_rst(means, stds)
    else:
        print("nothing to evaluate")

    if len(test_rsts)>0:
        means, stds = np.mean(test_rsts, 0), np.std(test_rsts, 0)
        print("{} seeds".format(len(test_rsts)))
        print_rst(means, stds)
    else:
        print("nothing to evaluate")
    print("-" * 50, "\n")
