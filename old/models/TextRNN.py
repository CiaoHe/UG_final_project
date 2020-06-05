# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, embedding, num_epochs, batch_size):
        self.model_name = 'DPCNN'
        self.vocab_path = './data/vocab.pkl'                                # 词表
        self.embedding_pretrained = torch.tensor(
            np.load('./data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.num_classes = 46                                           # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = num_epochs                                    # epoch数
        self.batch_size = batch_size                                    # mini-batch大小
        self.pad_size = 150                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-2                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 8                                             # lstm层数

'''Recurrent Neural Network for Text Classification with Multi-Task Learning'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.bn_0 = nn.BatchNorm1d(config.embed)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.bn = nn.BatchNorm1d(config.hidden_size * 2)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        #bn_0
        out = out.permute(0,2,1)
        out = self.bn_0(out)
        out = out.permute(0,2,1)
        #lstm
        out, _ = self.lstm(out)
        #bn
        out = out.permute(0,2,1)
        out = self.bn(out)
        out = out.permute(0,2,1)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out