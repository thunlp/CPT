import json

import pdb


def simple_parse(v):
    """
        use hand writen rules to parse
    """
    q = v['question']
    ann = v['annotations']['question']
    # parse all pos
    all_pos = sorted([slice(int(k), int(k)+1) if ":" not in k else slice(int(k.split(":")[0]), int(k.split(":")[1])) for k in ann],
                     key=lambda x: x if type(x) is int else x.start)
    words = q.replace("?", "").split()
    print([words[p] for p in all_pos])

    # if only one pos, just return
    if len(all_pos) <= 1:
        return " ".join(words)+"?"
    # if "piece of furniture to the left of the plate" in q:
    #     pdb.set_trace()
    # merge multiple words into single referring expression
    # e.g. the man wearing black shirt -> the man
    selection_mask = [True for w in words]
    # for other
    for i in range(len(all_pos)-1, 0, -1):
        now, previous = all_pos[i], all_pos[i-1]
        if "that" == words[previous.stop]:
            for j in range(previous.stop, now.stop):
                selection_mask[j] = False
        elif "and" in words[previous.stop: now.start] or "or" in words[previous.stop: now.start]:
            continue
        elif len(words) != now.stop:
            for j in range(previous.stop, now.stop):
                selection_mask[j] = False

    words = [w for w, flag in zip(words, selection_mask) if flag]
    return " ".join(words)+"?"
    # print(q, "|", [q.split()[pos] for pos in all_pos])


def _match(words, key_word):
    best_match = _match_predicate(words, key_word)
    if best_match != -1:
        return best_match
    max_matched_len = 0
    for i, w in enumerate(words):
        matched_len = -1
        if key_word in w:
            matched_len = len(key_word)
        if w in key_word:
            matched_len = len(w)
        if matched_len >= max_matched_len:
            best_match = i
            max_matched_len = matched_len
    return best_match


def _match_predicate(words, key_word):
    sentence = " ".join(words)
    idx = sentence.rfind(key_word)

    word_begins = []
    ptr = 0
    for w in words:
        word_begins.append(ptr)
        ptr += len(w)+1
    word_begins.append(1000)

    if idx == -1:
        return -1

    end = idx + len(key_word)-1
    for i, w in enumerate(word_begins):
        if word_begins[i] < end < word_begins[i+1]:
            return i
    return -1


def _eliminate(mask, l, r):
    for i in range(l, r):
        mask[i] = False


def _get_another_name(process, question, construction_process):
    if process['operation'] == "select":
        return process['argument'].split("(")[0].strip()
    elif process['operation'] == "relate":
        subj, predicate, obj = process['argument'].split(",")
        if "(" in subj:
            return obj
        elif "(" in obj:
            return subj
    elif "filter" in process['operation']:
        dependencies = process['dependencies']
        if len(dependencies) > 0:
            previous_process = construction_process[dependencies[0]]
            pre_word = previous_process['argument'].split("(")[0].strip()
            pre_idx = question.rfind(pre_word)
            cur_idx = question.rfind(process['argument'])
            if pre_idx > cur_idx:
                return pre_word
        return process['argument']
    else:
        print(process)
        assert False
    return None


def gt_parse(v):
    q = v['question']
    ann = v['annotations']['question']
    # parse all pos
    all_pos = sorted(
        [slice(int(k), int(k) + 1) if ":" not in k else slice(int(k.split(":")[0]), int(k.split(":")[1])) for k in ann],
        key=lambda x: x if type(x) is int else x.start)
    words = q.replace("?", "").split()
    # obj_words = [" ".join(words[p]) for p in all_pos]
    construction_process = v['semantic']
    selection_mask = [True for w in words]
    for c in construction_process[::-1]:
        op = c['operation']
        argument = c['argument']
        dependencies = c['dependencies']
        if op == "relate":
            # if "Are there any men to the right of the backpack" in q:
            #     import pdb
            #     pdb.set_trace()
            subj, predicate, obj = argument.split(',')
            d = dependencies[0]
            another_name = _get_another_name(construction_process[d], q, construction_process)
            if subj == "_" or obj == "_":
                continue
            if "(" in subj:
                subj = another_name
            elif "(" in obj:
                obj = another_name
            obj_idx = _match(words, obj)
            subj_idx = _match(words, subj)
            pred_idx = _match_predicate(words, predicate)
            min_idx = min(obj_idx, subj_idx, pred_idx)
            max_idx = max(obj_idx, subj_idx, pred_idx)
            if obj_idx == subj_idx:
                continue
            if min_idx == -1:
                continue
            _eliminate(selection_mask, min_idx+1, max_idx+1)
    words = [w for w, flag in zip(words, selection_mask) if flag]
    return " ".join(words) + "?"


# values = []
# for i in range(4):
#     l = json.load(open("/data_local/zhangao/data/gqa/train_all_questions/train_all_questions_{}.json".format(i)))
#     n = 0
#     import numpy as np
#     np.random.seed(0)
#     tmp = np.random.choice(list(l.values()), 100)
#     tmp = list(tmp)
#     values.extend(tmp)
# json.dump(list(values), open("tmp/gqa.json", "w"))
values = json.load(open("tmp/gqa.json"))
for v in values:
    # if "Is there any traffic light to the left of the vehicle in the center of the photo" in v['question']:
    new_q = gt_parse(v)
    print(v['semantic'])
    print(v['question'])
    print(new_q)
    print()
