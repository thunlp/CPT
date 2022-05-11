# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.

import argparse
import os
import os.path as op

import numpy as np
import base64
import torch
import torch.distributed as dist
from transformers.pytorch_transformers import BertTokenizer, BertConfig
import random
from oscar.utils.tsv_file import TSVFile
from oscar.modeling.modeling_bert import BertImgForPreTraining
from oscar.utils.logger import setup_logger
from oscar.utils.misc import (mkdir, set_seed)
from oscar.datasets.refcoco_fsl_cpt_dataset import RefcocoCPTDataset
from oscar.modeling.modeling_rec import REC_MLM_CPT
import pickle
import json
from tqdm import tqdm
import torch.optim as optim
from torch import nn
from transformers.pytorch_transformers.modeling_bert import BertLayerNorm
from oscar.utils.comm import all_gather, gather_on_master, reduce_dict
from oscar.utils.iou import computeIoU
from oscar.utils.optim_sched import get_lr_sched
from oscar.utils.save_model import save_model



def get_evaluate_file(predict_file):
    assert predict_file.endswith('.tsv')
    fpath = op.splitext(predict_file)[0]
    return fpath + '.eval.json'


def get_evaluate_method(predict_file):
    if 'nocaps' in op.basename(predict_file):
        return 'nocaps'
    else:
        return 'coco'


def restore_training_settings(args):
    if args.do_train:
        if not args.scst:
            return args
        checkpoint = args.model_name_or_path
    else:
        assert args.do_test or args.do_eval
        checkpoint = args.eval_model_dir
    # restore training settings, check hasattr for backward compatibility
    train_args = torch.load(op.join(checkpoint, 'training_args.bin'))
    if hasattr(train_args, 'max_seq_a_length'):
        if hasattr(train_args, 'scst') and train_args.scst:
            max_od_labels_len = train_args.max_seq_length - train_args.max_gen_length
        else:
            max_od_labels_len = train_args.max_seq_length - train_args.max_seq_a_length
        max_seq_length = args.max_gen_length + max_od_labels_len
        args.max_seq_length = max_seq_length
        logger.warning('Override max_seq_length to {} = max_gen_length:{} + od_labels_len:{}'.format(
                max_seq_length, args.max_gen_length, max_od_labels_len))


    override_params = ['max_seq_a_length', 'do_lower_case', 'add_od_labels',
            'max_img_seq_length']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param,
                    test_v, train_v))
                setattr(args, param, train_v)
    return args


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def ensure_init_process_group(local_rank=None, port=12345):
    # init with env
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    if world_size > 1 and not dist.is_initialized():
        assert local_rank is not None
        print("Init distributed training on local rank {}".format(local_rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl', init_method='env://'
        )
    return local_rank

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


def train_collate(batch):
    new_batch = list(zip(*batch))
    img_keys, img_feat_list, input_ids_list, input_mask_list, segment_ids_list, mask_token_pos, gts, colors, rects \
        = map(lambda x: [b for a in x for b in a], new_batch)
    rects = new_batch[-1]

    img_feat_list = torch.stack(img_feat_list, 0)
    input_ids_list = torch.stack(input_ids_list, 0)
    input_mask_list = torch.stack(input_mask_list, 0)
    segment_ids_list = torch.stack(segment_ids_list, 0)
    mask_token_pos = torch.tensor(mask_token_pos, dtype=torch.long)
    gts = torch.tensor(gts, dtype=torch.long)
    n = 100
    return new_batch[0][:n], (img_feat_list[:n], input_ids_list[:n], input_mask_list[:n], segment_ids_list[:n], mask_token_pos[:n], gts[:n])


def test_collate(batch):
    new_batch = list(zip(*batch))
    img_keys, img_feat_list, input_ids_list, input_mask_list, segment_ids_list, mask_token_pos, gts, colors, rects \
        = map(lambda x: [b for a in x for b in a], new_batch)
    rects = new_batch[-1]
    colors = new_batch[-2]

    img_feat_list = torch.stack(img_feat_list, 0)
    input_ids_list = torch.stack(input_ids_list, 0)
    input_mask_list = torch.stack(input_mask_list, 0)
    segment_ids_list = torch.stack(segment_ids_list, 0)
    mask_token_pos = torch.tensor(mask_token_pos, dtype=torch.long)
    gts = torch.tensor(gts, dtype=torch.long)
    return new_batch[0], (img_feat_list, input_ids_list, input_mask_list, segment_ids_list, mask_token_pos, colors, rects)

def build_test_dataset(data_file, tokenizer, args):
    if not op.isfile(data_file):
        data_file = op.join(args.test_dir, data_file)
        assert op.isfile(data_file), data_file
    return RefcocoCPTDataset(data_file, tokenizer=tokenizer, args=args)


def build_train_dataset(data_file, tokenizer, args):
    if not op.isfile(data_file):
        data_file = op.join(args.train_dir, data_file)
        assert op.isfile(data_file), data_file
    return RefcocoCPTDataset(data_file, tokenizer=tokenizer, is_train=True, args=args)

def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_data_loader(args, data_file, tokenizer, is_distributed=True, is_train=False):
    if is_train:
        dataset = build_train_dataset(data_file, tokenizer, args)
        shuffle = True
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        logger.info("Train with {} images per GPU.".format(images_per_gpu))
        logger.info("Total batch size {}".format(images_per_batch))
        logger.info("Total training steps {}".format(num_iters))
        collate_fn = train_collate
    else:
        dataset = build_test_dataset(data_file, tokenizer, args)
        shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size
        collate_fn = test_collate

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, sampler=sampler,
        batch_size=images_per_gpu,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return data_loader


def train_batch(args, config, train_loader, test_loader, model, optimizer, tokenizer, global_step=0):
    for step, (img_keys, batch) in enumerate(train_loader):
        model.train()
        batch = tuple(t.to(args.device) for t in batch)
        img_feat_list, input_ids_list, input_mask_list, segment_ids_list, mask_token_pos, colors = batch
        # construct mlm labels
        mlm_labels = torch.zeros(input_mask_list.size(), dtype=torch.long, device=args.device)
        mlm_labels[:] = -1
        mlm_labels[torch.arange(input_mask_list.size(0)), mask_token_pos] = colors
        # train
        # schedule lr
        lr_this_step = get_lr_sched(global_step, config)
        for i, param_group in enumerate(optimizer.param_groups):
            if i == 0 or i == 1:
                param_group['lr'] = lr_this_step * config.lr_mul
            elif i == 2 or i == 3:
                param_group['lr'] = lr_this_step
            else:
                raise ValueError()
        try:
            optimizer.zero_grad()
            loss, output = model(input_ids_list, segment_ids_list, input_mask_list, img_feats=img_feat_list,
                                 masked_lm_labels=mlm_labels)
            loss.backward()
            optimizer.step()
            global_step += 1
        except RuntimeError as e:
            logger.info("run time error at step {}, which is {}".format(step, str(e)))
            continue

    return global_step


def val(args, data_loader, model, tokenizer):
    model.eval()
    predictions = {}
    for step, (img_keys, batch) in enumerate(data_loader):
        img_feat_list, input_ids_list, input_mask_list, segment_ids_list, mask_token_pos \
            = tuple(t.to(args.device) for t in batch[:-2])
        colors, rects = batch[-2], batch[-1]
        # generate scores
        with torch.no_grad():
            output = model(input_ids_list, segment_ids_list, input_mask_list, img_feats=img_feat_list)[0]
        scores = output[torch.arange(output.size(0)), mask_token_pos]

        # assign rects
        color2id = lambda c: tokenizer.convert_tokens_to_ids(c)
        ptr = 0
        for i, (imgk, color_list, rect_list) in enumerate(zip(img_keys, colors, rects)):
            # imgk: '10803'
            # color_list: [['red', 'purple', 'green', 'yellow', 'blue'], ['red', 'purple']]
            # rect_list: [[anchor, rect1, rect2, rect3, rect4], [anchor, rect5]]
            collected_rects = []
            collected_scores = []
            for j, (cur_color_set, cur_rect_set) in enumerate(zip(color_list, rect_list)):
                # cur_color_set: ['red', 'purple', 'green', 'yellow', 'blue']
                # cur_rect_set: [anchor, rect1, rect2, rect3, rect4]
                assert len(cur_rect_set) == len(cur_rect_set)
                cur_color_id_set = color2id(cur_color_set + ["none"])
                cur_color_scores = scores[ptr][cur_color_id_set]
                ptr += 1
                # if j != 0:
                #     anchor_score = collected_scores[0][0]
                #     cur_rect_set = cur_rect_set[1:]
                #     cur_color_scores = cur_color_scores[1:] # * (anchor_score/cur_color_scores[0])
                collected_rects += cur_rect_set
                collected_scores.append(cur_color_scores[0:-1]/cur_color_scores[-1])
            # pass
            collected_scores = torch.cat(collected_scores, -1)
            max_idx = collected_scores.argmax()
            max_rect = collected_rects[int(max_idx)]

            predictions[imgk] = max_rect
        assert ptr == scores.size(0)
    pred_list = all_gather(predictions)
    for cur_preds in pred_list:
        for k, v in cur_preds.items():
            assert (k not in predictions) or (predictions[k] == v)
            predictions[k] = v

    miou = 0
    if is_main_process():
        gts = data_loader.dataset.anns_dic
        gts = {k: v['bbox'] for k, v in gts.items()}
        for k, p in predictions.items():
            assert p[2] > p[0] and p[3] > p[1]
            p = [p[0], p[1], p[2] - p[0] + 1, p[3] - p[1] + 1]
            iou = computeIoU(p, gts[k])
            miou += iou > 0.5
        logger.info("miou: {:.2f}".format(miou / len(predictions) * 100))
    return miou / len(predictions) * 100


def build_optimizer(model, opts):
    """ Re linear may get larger learning rate """
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = [(n, p) for n, p in model.named_parameters()
                       if 'classifier' not in n]
    param_top = [(n, p) for n, p in model.named_parameters()
                 if 'classifier' in n]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_top
                    if not any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate,
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_top
                    if any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate,
         'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=opts.learning_rate, betas=opts.betas)
    return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", default=False, type=bool, required=False,
                        help="Test or train the model.")
    parser.add_argument("--train_dir", default='datasets/coco_caption', type=str, required=False,
                        help="The input data dir with all required files.")
    parser.add_argument("--test_dir", default='datasets/coco_caption', type=str, required=False,
                        help="The input data dir with all required files.")
    parser.add_argument("--train_yaml", default='train.yaml', type=str, required=False,
                        help="yaml file for training.")
    parser.add_argument("--test_yaml", default='test.yaml', type=str, required=False,
                        help="yaml file for testing.")
    parser.add_argument("--val_yaml", default='val.yaml', type=str, required=False,
                        help="yaml file used for validation during training.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or model type.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--loss_type", default='sfmx', type=str,
                        help="Loss function types: support kl, x2, sfmx")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--max_seq_length", default=70, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded.")
    parser.add_argument("--max_seq_a_length", default=40, type=int,
                        help="The maximum sequence length for caption.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help= "Probability to mask input sentence during training.")
    parser.add_argument("--max_masked_tokens", type=int, default=3,
                        help="The max number of masked tokens per sentence.")
    parser.add_argument("--add_od_labels", default=False, action='store_true',
                        help="Whether to add object detection labels or not")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--max_img_seq_length", default=50, type=int,
                        help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int,
                        help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='frcnn', type=str,
                        help="Image feature type.")
    parser.add_argument("--tie_weights", default=False, action='store_true',
                        help="Whether to tie decoding weights to that of encoding")
    parser.add_argument("--freeze_embedding", default=False, action='store_true',
                        help="Whether to freeze word embeddings in Bert")
    parser.add_argument("--label_smoothing", default=0, type=float,
                        help=".")
    parser.add_argument("--drop_worst_ratio", default=0, type=float,
                        help=".")
    parser.add_argument("--drop_worst_after", default=0, type=int,
                        help=".")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--output_mode", default='classification', type=str,
                        help="output mode, support classification or regression.")
    parser.add_argument("--num_labels", default=2, type=int,
                        help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear or")
    parser.add_argument("--num_workers", default=8, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_train_epochs", default=40, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=-1,
                        help="Save checkpoint every X steps. Will also perform evaluatin.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each save_steps.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="For distributed training.")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")
    # for self-critical sequence training
    parser.add_argument('--scst', action='store_true', help='Self-critical sequence training')
    parser.add_argument('--sc_train_sample_n', type=int, default=5,
                        help="number of sampled captions for sc training")
    parser.add_argument('--sc_baseline_type', type=str, default='greedy',
                        help="baseline tyep of REINFORCE algorithm")
    parser.add_argument('--sc_beam_size', type=int, default=1,
                        help="beam size for scst training")
    parser.add_argument('--cider_cached_tokens', type=str, default='coco-train-words.p',
                        help="path to cached cPickle file used to calculate CIDEr scores")
    # for generation
    parser.add_argument("--eval_model_dir", type=str, default='',
                        help="Model directory for evaluation.")
    parser.add_argument('--max_gen_length', type=int, default=20,
                        help="max length of generated sentences")
    parser.add_argument('--output_hidden_states', action='store_true',
                        help="Turn on for fast decoding")
    parser.add_argument('--num_return_sequences', type=int, default=1,
                        help="repeating times per image")
    parser.add_argument('--num_beams', type=int, default=1, help="beam search width")
    parser.add_argument('--num_keep_best', type=int, default=1,
                        help="number of hypotheses to keep in beam search")
    parser.add_argument('--temperature', type=float, default=1,
                        help="temperature in softmax for sampling")
    parser.add_argument('--top_k', type=int, default=0,
                        help="filter distribution for sampling")
    parser.add_argument('--top_p', type=float, default=1,
                        help="filter distribution for sampling")
    parser.add_argument('--repetition_penalty', type=int, default=1,
                        help="repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)")
    parser.add_argument('--length_penalty', type=int, default=1,
                        help="beam search length penalty")
    # for Constrained Beam Search
    parser.add_argument('--use_cbs', action='store_true',
                        help='Use constrained beam search for decoding')
    parser.add_argument('--min_constraints_to_satisfy', type=int, default=2,
                        help="minimum number of constraints to satisfy")
    args = parser.parse_args()

    global logger

    # Setup CUDA, GPU & distributed training
    local_rank = ensure_init_process_group(local_rank=args.local_rank)
    args.local_rank = local_rank
    args.num_gpus = get_world_size()
    args.distributed = args.num_gpus > 1
    args.device = torch.device('cuda')
    synchronize()

    output_dir = args.output_dir
    mkdir(output_dir)

    logger = setup_logger("vlpretrain", output_dir, args.local_rank)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.num_gpus)
    set_seed(args.seed, args.num_gpus)
    args = restore_training_settings(args)

    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = BertConfig, BertImgForPreTraining, BertTokenizer
    checkpoint = args.eval_model_dir
    assert op.isdir(checkpoint)
    config = config_class.from_pretrained(checkpoint)
    config.output_hidden_states = args.output_hidden_states
    tokenizer = tokenizer_class.from_pretrained(checkpoint)
    logger.info("Evaluate the following checkpoint: %s", checkpoint)
    tmp = model_class.from_pretrained(checkpoint, config=config)
    config.max_img_seq_length = args.max_img_seq_length
    model = REC_MLM_CPT(config)
    model.copy_from_pretraining_model(tmp)
    if args.test_mode:
        del model
        model = REC_MLM_CPT.from_pretrained(checkpoint, config=config)
    model.to(args.device)

    # optimizer
    config.learning_rate = 3e-5
    config.weight_decay = 0.01
    config.betas = [0.9, 0.98]
    config.lr_mul = 1.0
    optimizer = build_optimizer(model, config)

    # distributed
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )


    train_loader = make_data_loader(args, "predictions.tsv", tokenizer, is_distributed=args.distributed, is_train=True)
    test_loader = make_data_loader(args, "predictions.tsv", tokenizer, is_distributed=args.distributed, is_train=False)

    if args.test_mode:
        acc = val(args, test_loader, model, tokenizer)
        logger.info("The accuracy is {}".format(acc))
        return

    global_step = 0
    best_acc = 0
    config.num_train_steps = args.num_train_epochs
    config.warmup_steps = int(config.num_train_steps/10)
    for epoch in range(config.num_train_steps):
        logger.info("epoch: {}".format(epoch))
        global_step = train_batch(args, config, train_loader, test_loader, model, optimizer, tokenizer, global_step=global_step)
        # if (epoch+1) % 10 == 0:
        #     acc = val(args, test_loader, model, tokenizer)
        #     if epoch >= config.num_train_steps//2 and acc >= best_acc:
        #         if is_main_process():
        #             logger.info("save best model at {} step, the acc is {}".format(global_step, acc))
        #         save_model(args, model, tokenizer, logger, save_mode="best")
        #         best_acc = acc
    # save
    save_model(args, model, tokenizer, logger, save_mode="final")


if __name__ == "__main__":
    main()
