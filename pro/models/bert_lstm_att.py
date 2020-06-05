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
        self.model_name = 'bert_lstm_att'
        self.models_dir = os.path.join(HOME_DIR, 'models')
        self.BERT_USING = model_PTM
        self.model_dir = os.path.join(self.models_dir, self.BERT_USING)              # model路

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.max_seq_len = 150
        self.num_classes = 46                                           # 类别数
        self.num_epochs = num_epochs                                           # epoch数
        self.batch_size = batch_size                                            # mini-batch大小
        self.learning_rate = 2e-5                                      # 学习率
        self.hidden_size = 1024 if self.BERT_USING == 'chinese_rbtl3_pytorch' else 768

        self.dropout = 0.1

        self.rnn_hidden = 1024
        self.num_layers = 2

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.model_dir) #(batch_size, seq_length, hidden_size)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers, 
                            bidirectional=True, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        
        if str(config.device) == 'cuda':
            self.w_omega = torch.zeros((config.rnn_hidden * config.num_layers, 
                            config.rnn_hidden * config.num_layers)).cuda()
            self.u_omega = torch.zeros(
                (config.rnn_hidden * config.num_layers, 1)
            ).cuda()
        else:
            self.w_omega = torch.zeros((config.rnn_hidden * config.num_layers, 
                            config.rnn_hidden * config.num_layers))
            self.u_omega = torch.zeros(
                (config.rnn_hidden * config.num_layers, 1)
            ) 

        self.fc_nn = nn.Linear(config.rnn_hidden*2, config.num_classes)

    
    def attention_net(self, lstm_output):
        #lstm_output: [batch_size, seq_length, rnn_hidden_size*2]
        scores = torch.tanh(torch.matmul(lstm_output, self.w_omega))
        #scores: [B, L, rnn_hidden*2]
        attention_weights = F.softmax(torch.matmul(scores, self.u_omega), dim=1)
        #attetion_weights: [B, L, 1]
        scored_outputs = scores * attention_weights
        #scored_outputs: [B, L, rnn_hidden*2]

        outputs = torch.sum(scored_outputs, 1)
        #outputs: [B, rnn_hidden*2]
        return outputs


    
    def forward(self, x):
        seqs = x[0]
        seqs_masks = x[1]
        seq_segments = x[2]
        labels = x[3]

        encoder_out = self.bert(seqs, seqs_masks, seq_segments)[0]
        lstm_out, _ = self.lstm(encoder_out) #lstm_out: [batch_size, seq_length, rnn_hidden*2]
        attn_out = self.attention_net(lstm_out) #attn_out: [B, rnn_hidden*2]
        out = self.fc_nn(attn_out)
        return out
