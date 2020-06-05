# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)


'''Deep Pyramid Convolutional Neural Networks for Text Categorization'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.embed), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)       # [B, 1, L, H]
        x = self.conv_region(x)  # [B, 256, seq_len-3+1, 1]

        x = self.padding1(x)  # [B, 256, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [B, 256, seq_len-3+1, 1]
        x = self.padding1(x)  # [B, 256, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [B, 256, seq_len-3+1, 1]
        while x.size()[2] > 1:
            x = self._block(x)
        x = x.squeeze()  # [B, num_filters(256)]
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

        # Short Cut
        x = x + px
        return x