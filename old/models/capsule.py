# coding: UTF-8
import torch
import torch as t 
import torch.nn as nn
import torch.nn.functional as F 
import os
import numpy as np
import copy

class Config(object):

    """initialize parameters"""
    def __init__(self, embedding, num_epochs, batch_size):
        self.model_name = 'capsule'
        self.vocab_path = './data/vocab.pkl' 
        self.embedding_pretrained = torch.tensor(
            np.load('./data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None  

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.pad_size = 150
        self.num_classes = 46                                           # 类别数
        self.num_epochs = num_epochs                                    # epoch数
        self.batch_size = batch_size                                    # mini-batch大小
        self.learning_rate = 1e-3                                      # 学习率
        # self.hidden = 1024 
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度

        #params for capsule_net
        self.Routings = 5
        self.Num_capsule = 10
        self.Dim_capsule = 16
        self.T_epsilon = 1e-7

        #params for gru_layer
        self.gru_len = 256

class GRU_layer(nn.Module):
    def __init__(self, config):
        super(GRU_layer, self).__init__()
        self.gru = nn.GRU(input_size = config.embed, 
                          hidden_size = config.gru_len, 
                          bidirectional = True)
    
    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for k in ih:
            nn.init.xavier_uniform_(k)
        for k in hh:
            nn.init.xavier_uniform_(k)
        for k in b:
            nn.init.constant_(k, 0)
    
    def forward(self, x):
        # gru input:[B, L, H(300)], output(B, L, 2*gru_len(256))
        return self.gru(x)

class Caps_Layer(nn.Module):
    def __init__(self, config, share_weights=True, activation='default', **kwargs):
        super(Caps_Layer, self).__init__(**kwargs)
        self.input_dim_capsule = config.gru_len * 2   #2H
        self.num_capsule = config.Num_capsule   #num_capsule: 下一层的neuron数目,对应于j
        self.dim_capsule = config.Dim_capsule   #dim_capsule: neuron维度,对应于k
        self.routings = config.Routings         #routing迭代次数
        self.share_weights = share_weights
        self.T_epsilon = config.T_epsilon
        if activation == 'default':
            self.activation = self.squash
        else:
            self.activation = nn.ReLU(inplace=True)
        
        if self.share_weights:
            self.W = nn.Parameter(
                nn.init.xavier_normal_(t.empty(1, self.input_dim_capsule, self.num_capsule * self.dim_capsule))
            )
        else:
            self.W = nn.Parameter(
                t.randn(config.batch_size, self.input_dim_capsule, self.num_capsule * self.dim_capsule)
            )
    
    def forward(self, x):
        if self.share_weights:
            u_hat_vecs = t.matmul(x, self.W) #[B, L, num_cap*dim_cap]
        else:
            self.W = nn.Parameter(
                t.randn(x.shape[0], self.input_dim_capsule, self.num_capsule * self.dim_capsule)
            )
            u_hat_vecs = t.matmul(x, self.W) #[B, L, num_cap*dim_cap]
        
        batch_size = x.shape[0] #B
        input_num_capsule = x.shape[1]        #i 输入层的neuron数目
        u_hat_vecs = u_hat_vecs.view(batch_size, input_num_capsule, self.num_capsule, self.dim_capsule) #[B, i, j, k]
        u_hat_vecs = u_hat_vecs.permute(0,2,1,3)           #[B, j, i, k]
        b = t.zeros_like(u_hat_vecs[:,:,:,0]) #随机初始化b, [B, j, i]

        for r in range(self.routings):
            c = F.softmax(b, dim=1) #[B,j,i]
            v = self.activation(t.einsum('bji,bjik->bjk', (c, u_hat_vecs))) #[B, j, k] 
            
            if r<self.routings-1:
                b = t.einsum('bjk,bjik->bji',(v, u_hat_vecs))  #[B,j,i]

        return v

    def squash(self, x, axis = -1):
        s_squared_norm = (x**2).sum(axis, keepdim = True)
        scale = t.sqrt(s_squared_norm + self.T_epsilon)
        return x/scale

class Dense_Layer(nn.Module):
    def __init__(self, config):
        super(Dense_Layer, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(p=0, inplace = True),
            nn.Linear(config.Num_capsule*config.Dim_capsule, config.num_classes),
        )
        self.batch_size = config.batch_size
    
    def forward(self, x):
        #x:[B,i,k]->[B, i*k]
        x = x.view(x.shape[0], -1) 
        # try:
        #     x = self.fc(x)
        # except RuntimeError:
        #     print(x.shape)
        x = self.fc(x)
        return x


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        self.gru_layer = GRU_layer(config)
        self.gru_layer.init_weights()
        self.caps_layer = Caps_Layer(config)
        self.dense_layer = Dense_Layer(config)
    
    def forward(self, x):
        encoder_out = self.embedding(x) #[B, L, H]
        #print(encoder_out.shape)
        x, _ = self.gru_layer(encoder_out) #[B, L, 2*gru_len]
        x = self.caps_layer(x)             #[B, i, k]
        output = self.dense_layer(x)       #[B, num_classes]
        return output
