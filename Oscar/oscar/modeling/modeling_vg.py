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
                                                             BertPredictionHeadTransform, BertOnlyMLMHead,
                                                             BertLMPredictionHead,
                                                             BertConfig, BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
                                                             load_tf_weights_in_bert)

logger = logging.getLogger(__name__)


class VGFT(BertPreTrainedModel):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """

    def __init__(self, config):
        super(VGFT, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        if config.img_feature_dim > 0:
            self.bert = BertImgModel(config)

        self.classifier = nn.Linear(config.hidden_size * 2, 51)

        self.classifier.apply(self.init_weights)

        # loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, img_feats=None, pairs=None, rel_labels=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)
        sequence_output = outputs[0]

        img_hidden = self._get_image_hidden(sequence_output, pairs)
        logits = self.classifier(img_hidden)
        loss = None
        if self.training:
            loss = self.criterion(logits, rel_labels)
        return logits, loss

    def _get_image_hidden(self, sequence_output, pairs):
        """
        Extracting the img_hidden part from sequence_output.
        Inputs:
        - sequence_output: (n, txt_len+num_bb, hid_size)
        - txt_lens       : [txt_len]
        - num_bbs        : [num_bb]
        Output:
        - img_hidden     : (n, max_num_bb, hid_size)
        """
        sequence_output = sequence_output[:, 70:, :]
        pairs = [[[i] + p for p in p_list] for i, p_list in enumerate(pairs)]
        pairs = [y for x in pairs for y in x]
        pairs = torch.tensor(pairs, dtype=torch.long).to(sequence_output.device)
        subs = sequence_output[pairs[:, 0], pairs[:, 1]]
        objs = sequence_output[pairs[:, 0], pairs[:, 2]]
        img_hidden = torch.cat([subs, objs], 1)
        return img_hidden


