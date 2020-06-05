# coding: UTF-8
import torch
import torch.nn as nn
import os

from transformers import *

HOME_DIR = '/root/pro'

class Config(object):

    """initialize parameters"""
    def __init__(self, num_epochs, batch_size, model_PTM):
        self.model_name = 'xlnet'
        self.models_dir = os.path.join(HOME_DIR, 'models')
        self.BERT_USING = model_PTM
        self.model_dir = os.path.join(self.models_dir, self.BERT_USING)              # model路

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.max_seq_len = 150
        self.num_classes = 46                                           # 类别数
        self.num_epochs = num_epochs                                    # epoch数
        self.batch_size = batch_size                                    # mini-batch大小
        self.learning_rate = 2e-5                                       # 学习率
        self.hidden_size = 768

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.embed = XLNetModel.from_pretrained(config.model_dir)
        self.fc_nn = nn.Linear(config.hidden_size, config.num_classes)
    
    def forward(self, x):
        seqs = x[0]
        seq_masks = x[1]
        seq_segments = x[2]
        labels = x[3]
        last_hidden_states = self.embed(seqs, attention_mask = seq_masks)[0]
        pooled = last_hidden_states[:,-1,:].squeeze(1)
        # print(pooled.shape)
        out = self.fc_nn(pooled)
        return out