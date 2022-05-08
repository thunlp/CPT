import glob
import sys
import os
import numpy as np
import pickle

root = sys.argv[1]
alpha = float(sys.argv[2])
paths = glob.glob(os.path.join(root, "*"))

def get_acc(path):
    l=pickle.load(open(path, "rb"))
    l1=pickle.load(open(path.replace("pt", "cpt"), "rb"))
    dic1 = {d["questionId"]:d for d in l1}
    correct = 0
    if "vcr_qar" in path:
        for v in l:
            v1 = dic1[v["questionId"]]
            pred_ans = (np.array(v["logits"][0])*alpha + np.array(v1["logits"][0])).argmax()
            pred_rat = (np.array(v["logits"][1])*alpha + np.array(v1["logits"][1])).argmax()
            correct += (pred_ans==v["gt"][0])&(pred_rat==v["gt"][1])
    else:
        for v in l:
            v1 = dic1[v["questionId"]]
            pred = (np.array(v["logits"])*alpha + np.array(v1["logits"])).argmax()
            correct += (pred==v["gt"])

    return correct/len(l)

paths = sorted(paths, key=lambda x: int(x.split("/")[-1]))
for p in paths:
    shot = int(p.split("/")[-1])
    pks = glob.glob(os.path.join(p, "*", "*.pk"))
    pks = [pk for pk in pks if int(pk.split("/")[-2])>-1]
    accs=[get_acc(pk) for pk in pks]
    #print(shot, np.mean(accs), np.std(accs))
    print(shot, round(np.mean(accs)*100, 2), "+_", round(np.std(accs)*100, 2))


