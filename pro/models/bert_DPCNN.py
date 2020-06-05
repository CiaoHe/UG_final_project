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
        self.model_name = 'bert_DPCNN'
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
    

class Model(nn.Module):

    def __init__(self,config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.model_dir)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.hidden_size), stride = 1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride = 1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3,1), stride=2)
        self.padding1 = nn.ZeroPad2d((0,0,1,1))
        self.padding2 = nn.ZeroPad2d((0,0,0,1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def forward(self, x):
        seqs = x[0]
        seqs_masks = x[1]
        seq_segments = x[2]
        labels = x[3]

        encoder_out = self.bert(seqs, seqs_masks, seq_segments)[0]
        x = encoder_out.unsqueeze(1) # [batch_size, 1, seq_len, embed]
        x = self.conv_region(x) # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x) # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x) # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x) # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 1:
            x = self._block(x)
        x = x.squeeze() # [batch_size, num_filters(256)]
        x = self.fc(x)

        return x
        
    
    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)
        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)
        x = x + px  # short cut
        return x