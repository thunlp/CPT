import os
import time
import json
import logging
import random
import glob
import base64
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from oscar.utils.tsv_file import TSVFile
from oscar.utils.misc import load_from_yaml_file
from oscar.utils.iou import computeIoU


class NormalFinetuneDataset(Dataset):
    def __init__(self, data_file, args=None, tokenizer=None, txt_seq_len=70, img_seq_len=50, shuffle=False, **kwargs):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.txt_seq_len = txt_seq_len
        self.img_seq_len = img_seq_len
        self.corpus_tsvfile = TSVFile(data_file)
        ann_name = "finetune_" + os.path.basename(os.path.dirname(data_file)) + ".json"
        root_path = data_file.split("output")[0] + "data/refcoco/split/"
        ann_path = os.path.join(root_path, ann_name)
        self.anns_dic = json.load(open(ann_path))
        self.anns_dic = {str(d["id"]): d for d in self.anns_dic}

        self.shuffle = shuffle

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence".
        return len(self.corpus_tsvfile)

    def __getitem__(self, item):
        # load data
        img_name, od_labels, img_feat, caption, _, rects = self.decode_features(self.corpus_tsvfile, item)
        assert len(rects) == len(img_feat)

        if self.shuffle:
            rand_inds = [i for i in range(len(rects))]
            random.shuffle(rand_inds)
            rects = [rects[i] for i in rand_inds]
            od_labels = [od_labels[i] for i in rand_inds]
            img_feat = img_feat[rand_inds]
        od_labels = od_labels[: self.img_seq_len]
        od_labels = " ".join(od_labels)
        # od_labels = None

        img_feat = img_feat[: self.img_seq_len]
        rects = rects[: self.img_seq_len]

        # generate features
        input_ids, input_mask, segment_ids, lm_label_ids = tokenize(self.tokenizer,
                                                                    text_a=caption, text_b=od_labels, img_feat=img_feat,
                                                                    max_img_seq_len=self.img_seq_len,
                                                                    max_seq_a_len=40, max_seq_len=70,
                                                                    cls_token_segment_id=0,
                                                                    pad_token_segment_id=0, sequence_a_segment_id=0,
                                                                    sequence_b_segment_id=1)
        img_feat_len = len(img_feat)
        img_feat = torch.cat([img_feat, torch.zeros([self.img_seq_len-img_feat.size(0), 2054])], 0)

        # find correct rects
        gt_bbox = self.anns_dic[str(img_name)]['bbox']
        answer_tensor = torch.zeros(len(rects), dtype=torch.float)
        for i, p in enumerate(rects):
            p = [p[0], p[1], p[2] - p[0] + 1, p[3] - p[1] + 1]
            iou = computeIoU(gt_bbox, p)
            # if iou > 0.5:
            answer_tensor[i] = iou
        return img_name, img_feat, input_ids, input_mask, segment_ids, answer_tensor

    def decode_features(self, feat_tsv, img_idx):
        img_name, feat_str = feat_tsv.seek(img_idx)
        feat_info = json.loads(feat_str)
        objs, caption, _, _ = feat_info["objects"]
        # decode features
        assert len(objs) == 1
        # objs:[[{'rect':, 'feature':}, ...], [{}, ...], ...]
        boxlist = objs[0]
        # boxlist: [{'rect':, 'feature':}, ...]

        # im_feats
        boxlist_feats = [np.frombuffer(base64.b64decode(o['feature']), np.float32) for o in boxlist]
        boxlist_feats = torch.Tensor(np.stack(boxlist_feats))
        im_feats = boxlist_feats

        # od_labels
        boxlist_labels = [o['class'] for o in boxlist]
        od_labels = boxlist_labels

        # bboxes
        boxlist_rects = [o['rect'] for o in boxlist]
        rects = boxlist_rects
        return img_name, od_labels, im_feats, caption, None, rects


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def tokenize(tokenizer, text_a, text_b, img_feat, max_img_seq_len=50,
             max_seq_a_len=40, max_seq_len=70, cls_token_segment_id=0,
             pad_token_segment_id=0, sequence_a_segment_id=0, sequence_b_segment_id=1):
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenizer.tokenize(text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_len - 3)
    else:
        if len(tokens_a) > max_seq_len - 2:
            tokens_a = tokens_a[:(max_seq_len - 2)]

    t1_label = len(tokens_a)*[-1]
    if tokens_b:
            t2_label = [-1] * len(tokens_b)

    # concatenate lm labels and account for CLS, SEP, SEP
    if tokens_b:
        lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])
    else:
        lm_label_ids = ([-1] + t1_label + [-1])

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        assert len(tokens_b) > 0
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len
    assert len(lm_label_ids) == max_seq_len

    # image features
    if max_img_seq_len > 0:
        img_feat_len = img_feat.shape[0]
        if img_feat_len > max_img_seq_len:
            input_mask = input_mask + [1] * img_feat_len
        else:
            input_mask = input_mask + [1] * img_feat_len
            pad_img_feat_len = max_img_seq_len - img_feat_len
            input_mask = input_mask + ([0] * pad_img_feat_len)

    lm_label_ids = lm_label_ids + [-1] * max_img_seq_len

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.long)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    lm_label_ids = torch.tensor(lm_label_ids, dtype=torch.long)
    return input_ids, input_mask, segment_ids, lm_label_ids