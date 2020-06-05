# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from transformers import *

HOME_DIR = '.'

class Config(object):

    """initialize parameters"""
    def __init__(self, num_epochs, batch_size, model_PTM):
        self.model_name = 'xlnet_att'
        self.models_dir = os.path.join(HOME_DIR, 'models')
        self.BERT_USING = model_PTM
        self.model_dir = os.path.join(self.models_dir, self.BERT_USING)              # model路

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.max_seq_len = 150
        self.num_classes = 46                                           # 类别数
        self.num_epochs = num_epochs                                    # epoch数
        self.batch_size = batch_size                                    # mini-batch大小
        self.learning_rate = 2e-5                                       # 学习率
        self.hidden_size = 1024 if self.BERT_USING == 'chinese_rbtl3_pytorch' else 768
        self.dropout = 0.1

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.embed = XLNetModel.from_pretrained(config.model_dir) #(batch_size, seq_length, hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        if config.device == 'cuda':
            self.w_omega = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size)).cuda()
            self.u_omega = nn.Parameter(torch.Tensor(config.hidden_size, 1)).cuda()
        else:
            self.w_omega = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
            self.u_omega = nn.Parameter(torch.Tensor(config.hidden_size, 1))
        
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

        self.fc_nn = nn.Linear(config.hidden_size, config.num_classes)

    
    def attention_net(self, x):
        #x: [B, L, H]
        scores = torch.tanh(torch.matmul(x, self.w_omega))
        #scores: [B, L, H]
        attention_weights = F.softmax(torch.matmul(scores, self.u_omega), dim=1)
        #attetion_weights: [B, L, 1]
        scored_outputs = scores * attention_weights
        #scored_outputs: [B, L, H]

        outputs = torch.sum(scored_outputs, 1)
        #outputs: [B, H]
        return outputs


    
    def forward(self, x):
        seqs = x[0]
        seqs_masks = x[1]
        seq_segments = x[2]
        labels = x[3]

        encoder_out = self.embed(seqs, attention_mask = seqs_masks)[0]
        attn_out = self.attention_net(encoder_out) #attn_out: [B, H]
        out = self.fc_nn(attn_out)

        return out
