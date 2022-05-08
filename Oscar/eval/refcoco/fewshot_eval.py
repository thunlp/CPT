import sys
import os
import glob
import numpy as np

def ext_acc(p):
    with open(p, "r") as f:
        acc = 0
        for line in f.readlines():
            line = line.strip()
            if "The accuracy is" in line:
                line = line.split("The accuracy is ")[-1]
                if float(line) > 0:
                    acc = float(line)
    return acc

def get_str(l):
    accs = []
    l = sorted(l, key=lambda x: int(x.split("/")[-1]))
    l = l[0:5]
    for p in l:
        x = ext_acc(p)
        if x < 1:
            x*=100
        accs.append(x)
    return "acc is {:.2f}, std is {:.2f}".format(np.mean(accs), np.std(accs))

for i in [0, 16, 8, 4, 2, 1]:
    i = str(i)
    color_root = os.path.join(sys.argv[1], str(i))
    if not os.path.exists(color_root):
        continue

    print("shot:", int(i))

    splits = sorted(os.listdir(color_root))
    splits = ['refcoco_val', 'refcoco_testA', 'refcoco_testB', 'refcoco+_val', 'refcoco+_testA', 'refcoco+_testB','refcocog_val', 'refcocog_test']
    for sp in splits:
        if not os.path.exists(os.path.join(color_root, sp)):
            continue
        l = glob.glob(os.path.join(color_root, sp, "*"))
        print(sp, "\t", get_str(l))
    print()
    print("---------------------------------------------")
    



