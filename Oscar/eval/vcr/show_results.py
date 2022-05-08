import glob
import sys
import os
import numpy as np
import pickle

root = sys.argv[1]
paths = glob.glob(os.path.join(root, "*"))

def get_acc(path):
    l=pickle.load(open(path, "rb"))
    #return sum([x['gt']==np.array(x["logits"]).argmax() for x in l])/len(l)
    return sum([x['correct'] for x in l])/len(l)

paths = sorted(paths, key=lambda x: int(x.split("/")[-1]))
for p in paths:
    shot = int(p.split("/")[-1])
    pks = glob.glob(os.path.join(p, "*", "*.pk"))
    #print(pks)
    pks = [pk for pk in pks if int(pk.split("/")[-2])>-1]
    accs=[get_acc(pk) for pk in pks]
    #print(shot, accs)
    print(shot, round(np.mean(accs)*100, 2), "+_", round(np.std(accs)*100, 2))


