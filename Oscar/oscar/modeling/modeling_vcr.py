# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.
 
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math
import torch
from torch import nn
import copy
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


class NSPFT(BertPreTrainedModel):
    def __init__(self, config):
        super(NSPFT, self).__init__(config)

        self.bert = BertImgModel(config)
        self.classifier = nn.Sequential(
            # nn.Linear(config.hidden_size, config.hidden_size),
            # GELU(),
            # BertLayerNorm(config.hidden_size, eps=1e-12),
            nn.Linear(config.hidden_size, 2)
        )
        self.num_seq_relations = config.num_contrast_classes if hasattr(config, "num_contrast_classes") else 2

        self.apply(self.init_weights)

    def copy_from_pretraining_model(self, model, possible_colors=[]):
        self.bert = model.bert

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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None,
                position_ids=None, head_mask=None, img_feats=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)

        sequence_output, pooled_output = outputs[:2]
        seq_relationship_score = self.classifier(pooled_output)
        outputs = (seq_relationship_score, ) + outputs[2:]  # add hidden states and attention if they are here

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            outputs = (next_sentence_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)


class NSPCPT(BertPreTrainedModel):
    def __init__(self, config):
        super(NSPCPT, self).__init__(config)

        self.bert = BertImgModel(config)
        self.cls = BertLMPredictionHead(config)
        self.num_seq_relations = config.num_contrast_classes if hasattr(config, "num_contrast_classes") else 2

        self.apply(self.init_weights)
        self.tie_weights()

    def copy_from_pretraining_model(self, model, possible_colors=[]):
        self.bert = model.bert
        self.cls = model.cls.seq_relationship

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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None,
                position_ids=None, head_mask=None, img_feats=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)

        sequence_output, pooled_output = outputs[:2]
        seq_relationship_score = self.cls(pooled_output)
        outputs = (seq_relationship_score, ) + outputs[2:]  # add hidden states and attention if they are here

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, self.num_seq_relations), next_sentence_label.view(-1))
            outputs = (next_sentence_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)


class VCR_QAR_NSPFT(BertPreTrainedModel):
    def __init__(self, config):
        super(VCR_QAR_NSPFT, self).__init__(config)

        self.bert = BertImgModel(config)
        self.classifier_ans = nn.Sequential(
            nn.Linear(config.hidden_size, 2)
        )
        self.classifier_rat = nn.Sequential(
            nn.Linear(config.hidden_size, 2)
        )
        self.num_seq_relations = config.num_contrast_classes if hasattr(config, "num_contrast_classes") else 2

        self.apply(self.init_weights)

    def copy_from_pretraining_model(self, model, possible_colors=[]):
        self.bert = model.bert

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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None,
                position_ids=None, head_mask=None, img_feats=None, head=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)

        sequence_output, pooled_output = outputs[:2]
        seq_relationship_score = None
        if head == "ans":
            seq_relationship_score = self.classifier_ans(pooled_output)
        elif head == "rat":
            seq_relationship_score = self.classifier_rat(pooled_output)

        outputs = (seq_relationship_score, ) + outputs[2:]  # add hidden states and attention if they are here

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            outputs = (next_sentence_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)



class VCRQAR_NSPCPT(BertPreTrainedModel):
    def __init__(self, config):
        super(VCRQAR_NSPCPT, self).__init__(config)

        self.bert = BertImgModel(config)
        # self.cls = BertLMPredictionHead(config)
        self.cls_ans = None
        self.cls_rat = None
        self.num_seq_relations = config.num_contrast_classes if hasattr(config, "num_contrast_classes") else 2

        self.apply(self.init_weights)
        # self.tie_weights()

    def copy_from_pretraining_model(self, model, possible_colors=[]):
        self.bert = model.bert
        self.cls_ans = model.cls.seq_relationship
        self.cls_rat = copy.deepcopy(model.cls.seq_relationship)

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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None,
                position_ids=None, head_mask=None, img_feats=None, head=""):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)

        sequence_output, pooled_output = outputs[:2]
        seq_relationship_score = None
        if head == "ans":
            seq_relationship_score = self.cls_ans(pooled_output)
        elif head == "rat":
            seq_relationship_score = self.cls_rat(pooled_output)

        outputs = (seq_relationship_score, ) + outputs[2:]  # add hidden states and attention if they are here

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, self.num_seq_relations), next_sentence_label.view(-1))
            outputs = (next_sentence_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)