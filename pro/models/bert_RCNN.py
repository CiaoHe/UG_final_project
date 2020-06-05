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
        self.model_name = 'bert_RCNN'
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
        self.rnn_hidden = 256
        self.num_layers = 2
        self.dropout = 0.1

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.model_dir) #(batch_size, seq_length, hidden_size)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.max_seq_len)
        self.fc = nn.Linear(config.rnn_hidden * 2 + config.hidden_size, config.num_classes)
        

    def forward(self, x):
        seqs = x[0]
        seqs_masks = x[1]
        seq_segments = x[2]
        labels = x[3]

        encoder_out = self.bert(seqs, seqs_masks, seq_segments)[0]
        out,_ = self.lstm(encoder_out)
        out = torch.cat((encoder_out, out), 2)
        out = F.relu(out)
        out = out.permute(0,2,1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)

        return out