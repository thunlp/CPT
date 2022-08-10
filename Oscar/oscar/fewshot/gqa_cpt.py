# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import copy, time, json
import base64

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import _pickle as cPickle
from tqdm import tqdm
from oscar.modeling.modeling_bert import ImageBertForSequenceClassification
from transformers.pytorch_transformers import WEIGHTS_NAME, BertTokenizer, BertConfig
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from oscar.utils.tsv_file import TSVFile
from oscar.modeling.modeling_rec import REC_MLM_CPT
from oscar.modeling.modeling_bert import BertImgForPreTraining
from oscar.utils.comm import all_gather, gather_on_master, reduce_dict
import torch.distributed as dist

from oscar.utils.misc import set_seed
from oscar.utils.task_utils import (_truncate_seq_pair, convert_examples_to_features_vqa,
                        output_modes, processors)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertImgForPreTraining, BertTokenizer),
}


log_json = []
debug_size = 500


def _load_dataset(args, name):
    processor = processors[args.task_name]()
    labels = processor.get_labels(args.label_file)

    if name == 'train':
        examples = processor.get_train_examples(args.data_dir, 'gqa_all_qla_train_sub.json')
    elif name == 'val':
        examples = processor.get_dev_examples(args.data_dir, 'gqa_bal_qla_testdev.json')
    elif name == 'test':
        examples = processor.get_test_examples(args.data_dir, 'vqa_test.json')
    elif name == 'test-dev':
        examples = processor.get_test_examples(args.data_dir, 'gqa_bal_qla_testdev.json')

    return examples, labels


class GQADataset(Dataset):
    """ GQA Dataset """

    def __init__(self, args, color_img_feat_file, name, tokenizer):
        super(GQADataset, self).__init__()
        assert name in ['train', 'val', 'test-dev', 'test',]

        self.args = args
        self.name = name

        # load image features
        self.img_feat_tsv = TSVFile(args.img_feat_file)
        self.color_img_feat_tsv = TSVFile(color_img_feat_file)
        self.qid2feat = self.cons_imgid2idx(self.color_img_feat_tsv)
        self.imgid2feat = self.cons_imgid2idx(self.img_feat_tsv)

        self.output_mode = output_modes[args.task_name]
        self.tokenizer = tokenizer

        self.examples, self.labels = _load_dataset(args, name)
        self.labels = [self.labels[i] for i in range(len(self.labels))]
        # self.examples = [example for example in self.examples if str(example.q_id) in self.qid2feat]
        self.eval_dic = {str(example.q_id): example.label for example in self.examples}
        if name == "train":
            self.examples = [example for example in self.examples if str(example.q_id) in self.qid2feat]
            import random
            random.seed(args.random_seed)
            self.examples = random.choices(self.examples, k=args.n_sample)
        if self.args.use_color == 0:
            self.qid2feat = None

        logger.info('%s Data Examples: %d' % (name, len(self.examples)))

    def cons_imgid2idx(self, feat_tsv):
        path = feat_tsv.tsv_file.replace("predictions.tsv", "imgid2idx.json")
        if os.path.exists(path):
            return json.load(open(path, "r"))
        dic = {}
        for i in range(len(feat_tsv)):
            idx, _ = feat_tsv.seek(i)
            dic[idx] = i
        json.dump(dic, open(path, "w"))
        return dic


    def tensorize_example(self, example, cls_token_at_end=False, pad_on_left=False,
                    cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                    sequence_a_segment_id=0, sequence_b_segment_id=1,
                    cls_token_segment_id=1, pad_token_segment_id=0,
                    mask_padding_with_zero=True):
        text_a, img_name, od_labels, img_feat, rects = self.get_img_feature(example)

        tokens_a = self.tokenizer.tokenize(text_a)

        tokens_b = None
        example.text_b = "[MASK]"
        if example.text_b:
            tokens_b = self.tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self.args.max_seq_length - 2:
                tokens_a = tokens_a[:(self.args.max_seq_length - 2)]

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.args.max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == self.args.max_seq_length
        assert len(input_mask) == self.args.max_seq_length
        assert len(segment_ids) == self.args.max_seq_length

        # image features

        if img_feat.shape[0] > self.args.max_img_seq_length:
            img_feat = img_feat[0:self.args.max_img_seq_length, ]
            if self.args.max_img_seq_length > 0:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
        else:
            if self.args.max_img_seq_length > 0:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
            padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)
            if self.args.max_img_seq_length > 0:
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])
                # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]

        if self.args.output_mode == "classification":
            if (example.label is None):
                label_id = [0]
                score = [0]
            elif len(example.label) == 0:
                label_id = [0]
                score = [0]
            else:
                label_id = [l for l in example.label]
                score = example.score
        elif self.args.output_mode == "regression":
            if len(example.label) == 0:
                label_id = 0
            else:
                label_id = float(example.label)
        else:
            raise KeyError(self.args.output_mode)

        new_scores = target_tensor(len(self.labels), label_id, score)

        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(segment_ids, dtype=torch.long),
                torch.tensor([label_id[0]], dtype=torch.long),
                torch.tensor(new_scores, dtype=torch.float),
                img_feat,
                torch.tensor([example.q_id], dtype=torch.long),
                (torch.tensor(input_ids, dtype=torch.long) == 103).nonzero(as_tuple=False).squeeze(1).tolist(),)

    def __getitem__(self, index):
        if self.args.load_fast:
            example = self.features[index]
        else:
            entry = self.examples[index]
            example = self.tensorize_example(entry,
                cls_token_at_end=bool(self.args.model_type in ['xlnet']), # xlnet has a cls token at the end
                cls_token=self.tokenizer.cls_token,
                sep_token=self.tokenizer.sep_token,
                cls_token_segment_id=2 if self.args.model_type in ['xlnet'] else 0,
                pad_on_left=bool(self.args.model_type in ['xlnet']), # pad on the left for xlnet
                pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0)
        return example

    def __len__(self):
        return len(self.examples)

    def get_img_feature(self, example):
        """ decode the image feature """
        qid = example.q_id
        img_id = example.img_key
        color_idx = None
        if self.qid2feat:
            color_idx = self.qid2feat.get(str(qid), None)
        if color_idx:
            img_name, feat_str = self.color_img_feat_tsv.seek(color_idx)
            feat_info = json.loads(feat_str)
            boxlist, meta_infos = feat_info["objects"]

            # question generation
            positions_and_colors = meta_infos[0]
            positions = [0] + [x[0][0] for x in positions_and_colors]
            colors = [x[1] for x in positions_and_colors]
            question = example.text_a
            new_question = []
            for i in range(len(positions)-1):
                l_pos, r_pos = positions[i], positions[i+1]
                new_question.append(question[l_pos: r_pos])
                new_question.append(colors[i] + " ")
            new_question.append(question[positions[-1]:])
            # print(example.text_a, " ".join(new_question) + "?")
            text_a = "".join(new_question)
        else:
            img_name, feat_str = self.img_feat_tsv.seek(self.imgid2feat[str(img_id)])
            feat_info = json.loads(feat_str)
            boxlist = feat_info["objects"]
            text_a = example.text_a
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
        return text_a, img_name, od_labels, im_feats, rects


def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def trim_batch(batch):
    """ new batch func
    :param batch:
    :return:
    """
    print('batch size', len(batch))

    batch_size = len(batch)
    batch_tensors = []
    for ele in batch[0]:
        print(ele.shape, ele.size())
        zero_tensor = torch.zeros(([batch_size] + list(ele.size())))
        batch_tensors.append(zero_tensor)

    for b_id, b in enumerate(batch):
        print(b_id, len(b))
        for ele_id, ele in enumerate(b):
            print(ele_id, ele.shape)
            batch_tensors[ele_id][b_id] = ele
    return batch_tensors


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """
    #if args.local_rank in [-1, 0]: tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, num_workers=args.workers, sampler=train_sampler, batch_size=args.train_batch_size) #, collate_fn=trim_batch)

    args.gradient_accumulation_steps = len(train_dataloader.dataset)//(args.per_gpu_train_batch_size * dist.get_world_size())
    args.gradient_accumulation_steps = max(args.gradient_accumulation_steps, 1)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total) # original

    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    #train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args.seed, args.n_gpu)  # Added here for reproductibility (even between python 2 and 3)

    best_score = 0
    best_model = {
        'epoch': 0,
        'model': copy.deepcopy(model), #model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }

    #eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True)

    labels = train_dataset.labels
    label_token_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(lb))[0] for lb in labels]
    label_token_ids = torch.tensor(label_token_ids, dtype=torch.long)
    answer_labels = [tokenizer.tokenize(lb)[0] for lb in labels]

    for epoch in range(int(args.num_train_epochs)):
    #for epoch in train_iterator:
        #epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        total_loss = 0
        train_score = 0
        total_norm = 0
        count_norm = 0

        if args.adjust_dp and epoch>=3:
            logger.info("change droput ratio {} to 0.3".format(args.drop_out))
            if hasattr(model, 'module'):
                model.module.dropout.p = 0.3
                model.module.bert.dropout.p = 0.3
                model.module.bert.embeddings.dropout.p = 0.3
            else:
                model.dropout.p = 0.3
                model.bert.dropout.p = 0.3
                model.bert.embeddings.dropout.p = 0.3

        if args.adjust_loss and epoch>=args.adjust_loss_epoch:
            logger.info("\t change loss type from kl to bce")
            model.loss_type = 'bce'

        # debug
        #epoch = 20
        #global_step = epoch*math.ceil(len(train_dataset)/(args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)))

        t_start = time.time()
        for step, raw_batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in raw_batch[:-1])

            # construct mlm labels
            mlm_labels = torch.zeros(batch[1].size(), dtype=torch.long, device=args.device)
            mlm_labels[:] = -1
            # import pdb
            # pdb.set_trace()
            idxs = (batch[0] == 103).nonzero(as_tuple=False).squeeze(1).tolist()
            idxs0, idxs1 = [x[0] for x in idxs], [x[1] for x in idxs]
            # labels = [answer_labels[x] for x in batch[3].squeeze()]
            mlm_labels[idxs0, idxs1] = label_token_ids[batch[3].squeeze()].to(mlm_labels.device)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'masked_lm_labels':        mlm_labels,
                      'img_feats':      None if args.img_feature_dim == -1 else batch[5]}
            loss, output = model(**inputs)

            # print(tokenizer.convert_ids_to_tokens(batch[0][1].tolist()), labels[int(batch[3].squeeze()[1])])

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                total_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                count_norm += 1

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:# Log metrics
                    if args.local_rank not in [-1, 0]:
                        torch.distributed.barrier()

                    if args.local_rank in [-1, 0] and args.evaluate_during_training:
                    #if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        logger.info("Epoch: %d, global_step: %d" % (epoch, global_step))
                        eval_result, eval_score, upper_bound = evaluate(args, model, eval_dataset, prefix=global_step)
                        if eval_score > best_score:
                            best_score = eval_score
                            best_model['epoch'] = epoch
                            best_model['model'] = copy.deepcopy(model)

                        logger.info("EVALERR: {}%".format(100 * best_score))

                    if args.local_rank == 0:
                        torch.distributed.barrier()

                    logging_loss = tr_loss

            #if args.max_steps > 0 and global_step > args.max_steps:
            #    epoch_iterator.close()
            #    break

        # evaluation
        logger.info("Epoch: %d" % (epoch))
        if epoch % args.eval_epoch == 0:
            if epoch != 0:
                eval_result, eval_score = evaluate(args, model, eval_dataset, tokenizer=tokenizer, prefix=global_step)
            else:
                eval_result = ""
                eval_score = 0.01
            if eval_score > best_score:
                best_score = eval_score
                best_model['epoch'] = epoch
                best_model['model'] = copy.deepcopy(model)
                #best_model['optimizer'] = copy.deepcopy(optimizer.state_dict())

        # save checkpoints
        if (args.local_rank in [-1, 0]) and (args.save_epoch>0 and epoch%args.save_epoch == 0) and (epoch>args.save_after_epoch):
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(epoch, global_step))
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            model_to_save = best_model['model'].module if hasattr(model, 'module') else best_model['model']  # Take care of distributed/parallel training

            save_num = 0
            while (save_num < 10):
                try:
                    logger.info("Saving model attempt: {}".format(save_num))
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    tokenizer.save_pretrained(output_dir)
                    break
                except:
                    save_num += 1
            logger.info("Saving model checkpoint {0} to {1}".format(epoch, output_dir))

        epoch_log = {'epoch': epoch, 'eval_score': eval_score, 'best_score':best_score}
        log_json.append(epoch_log)
        if args.local_rank in [-1, 0]:
            with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                json.dump(log_json, fp)

        logger.info("PROGRESS: {}%".format(round(100*(epoch + 1) / args.num_train_epochs, 4)))
        logger.info("EVALERR: {}%".format(100*best_score))

        t_end = time.time()
        logger.info('Epoch: %d, Train Time: %.3f' % (epoch, t_end - t_start))

        #if args.max_steps > 0 and global_step > args.max_steps:
        #    train_iterator.close()
        #    break

    if args.local_rank in [-1, 0]: # Save the final model checkpoint
        with open(args.output_dir + '/eval_logs.json', 'w') as fp:
            json.dump(log_json, fp)

        output_dir = os.path.join(args.output_dir, 'best-{}'.format(best_model['epoch']))
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        model_to_save = best_model['model'].module if hasattr(model, 'module') else best_model['model']  # Take care of distributed/parallel training

        save_num = 0
        while (save_num < 10):
            try:
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                tokenizer.save_pretrained(output_dir)
                break
            except:
                save_num += 1
        logger.info("Saving the best model checkpoint epoch {} to {}".format(best_model['epoch'], output_dir))

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset=None, tokenizer=None, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    #if args.n_gpu > 1: model = torch.nn.DataParallel(model) # debug: single-gpu or multi-gpus

    labels = eval_dataset.labels
    label_token_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(lb))[0] for lb in labels]
    label_token_ids = torch.tensor(label_token_ids, dtype=torch.long)
    answer_labels = [tokenizer.tokenize(lb)[0] for lb in labels]

    results = []
    t_start = time.time()
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]: os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, num_workers=args.workers, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        results = []

        for raw_batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        #for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in raw_batch[:-1])

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                          'img_feats':      None if args.img_feature_dim == -1 else batch[5]}
                mask_pos = raw_batch[-1][0]
                outputs = model(**inputs)[0]
                outputs = outputs[torch.arange(len(outputs)), mask_pos, :]
                logits = outputs[:, label_token_ids]
                preds = logits.argmax(1)
                # import pdb
                # pdb.set_trace()
                preds = [answer_labels[int(p)] for p in preds]
            assert len(preds) == len(batch[-1])
            for i, (qid, p, gt) in enumerate(zip(batch[-1], preds, batch[3])):
                qid = int(qid[0])
                igt = eval_dataset.eval_dic[str(qid)]
                gt = [labels[int(x)] for x in igt]
                correct = [x==p for x in gt]
                correct = max(correct)
                results.append({"answer": p, "question_id": qid, "gt":gt, "igt": igt, "correct": correct, "logits": logits[i].cpu().numpy()})
                # results.append({"answer": p, "question_id": qid, "correct": correct,})

        result_list = all_gather(results)

        if is_main_process():
            results = [y for x in result_list for y in x]
            results_dic = {k["question_id"]: k for k in results}
            acc = sum([x["correct"] for x in results_dic.values()]) / len(eval_dataset)
            logger.info("Eval Results:")
            logger.info("Eval Score: %.3f" % (100*acc))
            results_dic = list(results_dic.values())
            # results_dic = [{""} for v in results_dic]
            # for d in results_dic:
            #     del d["correct"]
            path = args.result_dir
            if not os.path.exists(path):
                os.makedirs(path)
            import pickle
            with open(os.path.join(path, 'val_results.pk'), 'wb') as f:
                pickle.dump(results_dic, f)

    t_end = time.time()
    logger.info('Eva Time Cost: %.3f' % (t_end - t_start))
    return results


def test(args, model, eval_dataset=None, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    label2ans = cPickle.load(open(args.label2ans_file, 'rb'))
    logger.info('label2ans: %d' % (len(label2ans)))

    results = []
    t_start = time.time()
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]: os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval
        logger.info("***** Running Test {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        for batch in eval_dataloader:
        #for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                          'labels':         None,
                          'img_feats':      None if args.img_feature_dim == -1 else batch[5]}
                outputs = model(**inputs)
                logits = outputs[0]

                val, idx = logits.max(1)
                #logger.info('idx: %s, batch[6]: %s' % (str(idx.shape), str(batch[6].shape)))

                for i in range(idx.size(0)):
                    #logger.info('idx: %d, batch: %d' % (idx[i].item(), batch[6][i].item()))
                    result = {}
                    result['question_id'] = batch[6][i].item()
                    result['answer'] = label2ans[eval_dataset.labels[idx[i].item()]] #label2ans[idx[i].item()]
                    results.append(result)

                    if len(results) % 2000 == 0:
                        logger.info("PROGRESS: {}%".format(round(100*len(results)/len(eval_dataset), 4)))
                    #logger.info('q_id: {0}, answer: {1}'.format(result['question_id'], result['answer']))

    with open(args.output_dir + ('/{}_results.json'.format(eval_dataset.name)), 'w') as fp:
        json.dump(results, fp)

    t_end = time.time()
    logger.info('# questions: %d' % (len(results)))
    logger.info('Test Time Cost: %.3f' % (t_end - t_start))


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    processor = processors[task]()
    output_mode = output_modes[task]

    label_list = processor.get_labels(args.label_file)

    t_start = time.time()
    examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)

    #img_features = torch.load(os.path.join(args.data_dir, 'val_img_feats.pt' if evaluate else 'train_img_feats.pt'))
    #img_features = torch.load(os.path.join(args.data_dir, 'val_img_frcnn_feats.pt' if evaluate else 'train_img_frcnn_feats.pt'))
    img_features = np.load(os.path.join(args.data_dir, 'val_img_frcnn_feats.npy' if evaluate else 'train_img_frcnn_feats.npy'))

    features = convert_examples_to_features_vqa(examples, img_features, label_list, args.max_img_seq_length, args.max_seq_length,
            tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)

    #if args.local_rank in [-1, 0]:
    #    logger.info("Saving features into cached file %s", cached_features_file)
    #    torch.save(features, cached_features_file)
    t_end = time.time()
    logger.info('Info: loading features using %.5f secs' % (t_end-t_start))


    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long) # batch*max_seq_len
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        labels = torch.tensor([f.label_id[0] for f in features], dtype=torch.long)
        targets = torch.tensor([target_tensor(len(label_list), f.label_id, f.score) for f in features], dtype=torch.float)

        if args.img_feature_dim > 0: # change here
            t_start = time.time()
            img_feat_np = np.zeros((labels.shape[0], args.max_img_seq_length, args.img_feature_dim))
            for f_id, f in enumerate(features):
                img_feat_np[f_id] = f.img_feat

            img_feats = torch.from_numpy(img_feat_np)

            #img_feats = torch.empty((labels.shape[0], args.max_img_seq_length, args.img_feature_dim))
            #for f_id, f in enumerate(features):
            #   img_feats[f_id] = f.img_feat

            t_end = time.time()
            logger.info('Info: convert image tensor features using %.5f secs' % (t_end - t_start))

            #img_feats = torch.stack([f.img_feat[:,-args.img_feature_dim:] for f in features])
            #img_feats = torch.stack([f.img_feat for f in features])
        #img_feats = img_feats.type(torch.long)

        #print('targets:', targets.shape)
        print('img_feats:', img_feats.shape)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    if args.img_feature_dim == -1:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, labels, targets)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, labels, targets, img_feats)
    return dataset

def target_tensor(len, labels, scores):
    """ create the target by labels and scores """
    target = [0]*len
    for id, l in enumerate(labels):
        target[l] = 1

    return target


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--label_file", type=str, default=None, help="Label Dictionary")
    parser.add_argument("--label2ans_file", type=str, default=None, help="Label to Answer Dictionary")

    parser.add_argument("--eval_epoch", type=int, default=1000, help="Label to Answer Dictionary")
    parser.add_argument("--result_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--img_feat_file", default=None, type=str, help="The input img_feat_file.")
    parser.add_argument("--train_color_img_feat_file", default=None, type=str, help="The input color_img_feat_file.")
    parser.add_argument("--testdev_color_img_feat_file", default=None, type=str, help="The input color_img_feat_file.")

    parser.add_argument("--qid2feat", default=None, type=str, help="The input qid2feat.")
    parser.add_argument("--n_sample", default=None, type=int, help="The input img_feat_file.")
    parser.add_argument("--random_seed", default=None, type=int, help="The input img_feat_file.")
    parser.add_argument("--use_color", default=None, type=int, help="The input img_feat_file.")

    parser.add_argument("--data_label_type", default='faster', type=str, help="faster or mask")
    parser.add_argument("--loss_type", default='kl', type=str, help="kl or xe")
    #parser.add_argument("--use_img_layernorm", action='store_true', help="use_img_layernorm")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train_val", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run test on the test set.")
    parser.add_argument("--do_test_dev", action='store_true', help="Whether to run test on the test-dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true', help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out for BERT.")
    parser.add_argument("--adjust_dp",action='store_true', help="Adjust Drop out for BERT.")

    parser.add_argument("--adjust_loss", action='store_true', help="Adjust Loss Type for BERT.")
    parser.add_argument("--adjust_loss_epoch", default=-1, type=int, help="Adjust Loss Type for BERT.")
    parser.add_argument("--classifier", default='linear', type=str, help="linear or mlp")
    parser.add_argument("--cls_hidden_scale", default=2, type=int, help="cls_hidden_scale: for classifier")

    parser.add_argument("--hard_label", action='store_true', help="Soft Label or Hard Label.")

    parser.add_argument("--max_img_seq_length", default=30, type=int, help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='faster_r-cnn', type=str, help="faster_r-cnn or mask_r-cnn")
    parser.add_argument("--code_voc", default=512, type=int, help="dis_code_voc: 256, 512")
    parser.add_argument("--code_level", default='top', type=str, help="code level: top, botttom, both")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=-1, help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_epoch', type=int, default=5, help="Save checkpoint every X epochs.")
    parser.add_argument('--save_after_epoch', type=int, default=-1, help="Save checkpoint after epoch.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true', help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument("--philly", action='store_true', help="Use Philly: reset the output dir")
    parser.add_argument("--load_fast", action='store_true', help="Load Tensor Fast")
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')

    #args = '--data_dir ../vqa/ban-vqa/data/qal_pairs --model_type bert --model_name_or_path bert-base-uncased --task_name vqa_text ' \
    #       '--do_train --do_eval --do_lower_case --max_seq_length 40 --per_gpu_eval_batch_size 16 --per_gpu_train_batch_size 16 --learning_rate 2e-5 ' \
    #       '--num_train_epochs 20.0 --output_dir ./results/vqa_text --label_file ../vqa/ban-vqa/data/cache/trainval_ans2label.pkl ' \
    #       '--save_steps 5000 --overwrite_output_dir --max_img_seq_length 1 --img_feature_dim 565 --img_feature_type dis_code '

    #args = '--data_dir ../vqa/ban-vqa/data/qal_pairs --model_type bert --model_name_or_path bert-base-uncased --task_name vqa_text ' \
    #       '--do_train --do_eval --do_lower_case --max_seq_length 40 --per_gpu_eval_batch_size 16 --per_gpu_train_batch_size 16 --learning_rate 2e-5 ' \
    #       '--num_train_epochs 20.0 --output_dir ./results/vqa_text --label_file ../vqa/ban-vqa/data/cache/trainval_ans2label.pkl ' \
    #       '--save_steps 5000 --overwrite_output_dir --max_img_seq_length 10 --img_feature_dim 565 --img_feature_type other '

    #args = parser.parse_args(args.split())

    args = parser.parse_args()

    if args.philly:  # use philly
        logger.info('Info: Use Philly, all the output folders are reset.')
        args.output_dir = os.path.join(os.getenv('PT_OUTPUT_DIR'), args.output_dir)
        logger.info('OUTPUT_DIR:', args.output_dir)

    #if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    #    raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train: logger.info("Output Directory Exists.")

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        logger.info("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed, args.n_gpu)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))

    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels(args.label_file)
    num_labels = len(label_list)
    logger.info('Task Name: {}, #Labels: {}'.format(args.task_name, num_labels))

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels, finetuning_task=args.task_name,
    )
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

    # discrete code
    config.img_feature_dim = args.img_feature_dim
    config.img_feature_type = args.img_feature_type
    config.code_voc = args.code_voc
    config.hidden_dropout_prob = args.drop_out
    config.loss_type = args.loss_type
    config.classifier = args.classifier
    config.cls_hidden_scale = args.cls_hidden_scale
    #config.use_img_layernorm = args.use_img_layernorm
    
    # load discrete code
    if args.img_feature_type in ['dis_code', 'dis_code_t']:
        logger.info('Load discrete code from: {}'.format(args.data_dir))
        t_start = time.time()
        train_code = torch.load(os.path.join(args.data_dir, 'vqvae', 'train.pt'))
        t_end = time.time()
        logger.info('Load time: %.3f' % (t_end - t_start))

        if args.code_level == 'top':
            config.code_dim = train_code['embeddings_t'].shape[0]
            config.code_size = train_code['feats_top'][list(train_code['feats_top'].keys())[0]].shape[0]
        elif args.code_level == 'bottom':
            config.code_dim = train_code['embeddings_b'].shape[0]
            config.code_size = train_code['feats_bottom'][list(train_code['feats_bottom'].keys())[0]].shape[0]
        elif args.code_level == 'both':
            config.code_dim = train_code['embeddings_t'].shape[0] + train_code['embeddings_b'].shape[0]

    tmp = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    model = REC_MLM_CPT(config)
    model.copy_from_pretraining_model(tmp)
    del  tmp

    if args.img_feature_type in ['dis_code', 'dis_code_t']:
        logger.info('Initializing the code embedding with {}'.format(args.code_level))
        if args.code_level == 'top':
            model.init_code_embedding(train_code['embeddings_t'].t())
        elif args.code_level == 'bottom':
            model.init_code_embedding(train_code['embeddings_b'].t())

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    #if args.do_eval:
    eval_dataset = GQADataset(args, args.testdev_color_img_feat_file, 'test-dev', tokenizer)
    # evaluate(args, model, eval_dataset, tokenizer, prefix="")

    # Training
    if args.do_train and (int(args.result_dir.split("/")[-2]) != 0):
        train_dataset = GQADataset(args, args.train_color_img_feat_file, 'train', tokenizer)
    #     # train_dataset = VCRDataset(args, 'val', img_features_path, tokenizer, label_pos_feats)
        train(args, train_dataset, eval_dataset, model, tokenizer)
    evaluate(args, model, eval_dataset, tokenizer, prefix="")

if __name__ == "__main__":
    main()
