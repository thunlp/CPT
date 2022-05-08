# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.
 
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, GELU, LayerNorm
from oscar.modeling.modeling_bert import BertImgModel, BertLMPredictionHead
from transformers.pytorch_transformers.modeling_bert import (BertEmbeddings,
        BertSelfAttention, BertAttention, BertEncoder, BertLayer,
        BertSelfOutput, BertIntermediate, BertOutput,
        BertPooler, BertLayerNorm, BertPreTrainedModel,
		BertPredictionHeadTransform, BertOnlyMLMHead, BertLMPredictionHead,
        BertConfig, BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        load_tf_weights_in_bert)

logger = logging.getLogger(__name__)


class REC_FT(BertPreTrainedModel):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """
    def __init__(self, config):
        super(REC_FT, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        if config.img_feature_dim > 0:
            self.bert = BertImgModel(config)

        # self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                GELU(),
                BertLayerNorm(config.hidden_size, eps=1e-12),
                nn.Linear(config.hidden_size, 1)
            )

        self.classifier.apply(self.init_weights)

        # loss
        self.loss_type = config.loss_type
        assert self.loss_type in ['cls', 'rank']
        if self.loss_type == 'rank':
            self.margin = config.margin
            self.hard_ratio = config.hard_ratio
        else:
            # self.crit = nn.CrossEntropyLoss(reduction='none')
            # self.criterion = nn.BCEWithLogitsLoss()
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
            position_ids=None, head_mask=None, img_feats=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)
        sequence_output = outputs[0]
        len_labels = [len(x) for x in labels]
        img_hidden = self._get_image_hidden(sequence_output, len_labels)
        logits = self.classifier(img_hidden).squeeze(1)
        # labels = torch.cat(labels, dim=0)

        # for CE
        max_num_bb = max(len_labels)
        targets = torch.tensor([t.argmax() for t in labels], device=labels[0].device).long()
        scores = torch.zeros([input_ids.size(0), max_num_bb], dtype=logits.dtype, device=logits.device).fill_(1e-4)
        ptr = 0
        for i, len_lb in enumerate(len_labels):
            scores[i, :len_lb] = logits[ptr: ptr+len_lb]
            ptr += len_lb
        assert ptr == sum(len_labels)

        loss = None
        if self.training:
            # loss = self.criterion(logits, labels.float())
            loss = self.criterion(scores, targets)

        logits = logits.split(len_labels)
        pred_idxs = [p.argmax() for p in logits]
        return loss, logits, pred_idxs

    def _get_image_hidden(self, sequence_output, len_labels):
        """
        Extracting the img_hidden part from sequence_output.
        Inputs:
        - sequence_output: (n, txt_len+num_bb, hid_size)
        - txt_lens       : [txt_len]
        - num_bbs        : [num_bb]
        Output:
        - img_hidden     : (n, max_num_bb, hid_size)
        """
        outputs = []
        for seq, len_lb in zip(sequence_output, len_labels):
            outputs.append(seq[70:70+len_lb])
        img_hidden = torch.cat(outputs, dim=0)
        return img_hidden


class REC_MLM_CPT(BertPreTrainedModel):
    def __init__(self, config):
        super(REC_MLM_CPT, self).__init__(config)

        self.bert = BertImgModel(config)
        self.cls = BertLMPredictionHead(config)
        self.num_seq_relations = config.num_contrast_classes if hasattr(config, "num_contrast_classes") else 2

        self.apply(self.init_weights)
        self.tie_weights()

    def copy_from_pretraining_model(self, model, possible_colors=[]):
        self.bert = model.bert
        self.cls = model.cls.predictions
        self.tie_weights()

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                position_ids=None, head_mask=None, img_feats=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)

        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores, ) + outputs[2:]  # add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)


