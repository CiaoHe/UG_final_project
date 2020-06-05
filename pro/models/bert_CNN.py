# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from transformers import *

HOME_DIR = '.'

class Config(object):

    """initialize parameters"""
    def __init__(self,num_epochs, batch_size, model_PTM):
        self.model_name = 'bert_CNN'
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

        self.filter_size = [2, 3, 4]                                    #卷积核尺寸
        self.num_filters = 256                                          #卷积核数量（channels数量）
        self.dropout = 0.1

class Model(nn.Module):
    
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.model_dir)   #(batch_size, seq_length, hidden_size)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList([ nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_size])
        # conv_output: (batch_size, num_filters, H, hidden_size), H:conved_seqLength
        self.dropout = nn.Dropout(config.dropout)
        self.fc_nn = nn.Linear(config.num_filters * len(config.filter_size), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  #x:(batch_size, num_filters, H*hiddden_size)
        x = F.max_pool1d(x, x.size(2)).squeeze(2) #x:(batch_size, num_filters)
        return x

    def forward(self, x):
        seqs = x[0]
        seqs_masks = x[1]
        seq_segments = x[2]
        labels = x[3]

        encoder_out = self.bert(seqs, seqs_masks, seq_segments)[0]
        out = encoder_out.unsqueeze(1)  #(batch_size, 1, seq_lengths, hidden_size), input_channels = 1
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], dim = 1)
        out = self.dropout(out)
        out = self.fc_nn(out) #out:(batch_size, num_classes)

        return out







