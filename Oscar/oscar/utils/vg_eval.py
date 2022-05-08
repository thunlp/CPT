import torch
import json
import pickle
from collections import Counter
import numpy as np


def eval_vg(preds, gts):
    imkey2pair = lambda n: [int(x) for x in n.split("_")[-2:]]

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
    print(len(new_preds))
    gts = [g for g in gts if g['img_path'] in new_preds]
    assert len(new_preds) == len(gts), "{}, {}".format(len(new_preds), len(gts))
    preds = [new_preds[k["img_path"]] for k in gts]

    recall = {20:[], 50: [], 100: []}
    mrecall = {20: [[] for i in range(51)], 50: [[] for i in range(51)], 100: [[] for i in range(51)]}

    for p_list, gt in zip(preds, gts):
        pairs = [p['pair'] for p in p_list]
        prds = [p['pred'] for p in p_list]
        pairs = torch.tensor(pairs)
        prds = torch.stack(prds, 0)
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
    rst = "R@20: {:.4f}\tR@50: {:.4f}\tR@100: {:.4f}".format(recall[20], recall[50], recall[100]) + "\n"
    rst += "mR@20: {:.4f}\tmR@50: {:.4f}\tmR@100: {:.4f}".format(mrecall[20], mrecall[50], mrecall[100]) + "\n"
    return rst
