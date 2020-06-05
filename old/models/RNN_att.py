# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config(object):

    """配置参数"""
    def __init__(self, embedding, num_epochs, batch_size):
        self.model_name = 'TextRCNN'
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
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 8                                             # lstm层数
        self.hidden_size2 = 64

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers, 
                            bidirectional=True, batch_first=True, dropout=config.dropout)

        if config.device == 'cuda':
            self.w_omega = nn.Parameter(torch.Tensor(config.hidden_size*2, config.hidden_size*2)).cuda()
            self.u_omega = nn.Parameter(torch.Tensor(config.hidden_size*2, 1)).cuda()
        else:
            self.w_omega = nn.Parameter(torch.Tensor(config.hidden_size*2, config.hidden_size*2))
            self.u_omega = nn.Parameter(torch.Tensor(config.hidden_size*2, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

        self.fc1 = nn.Linear(config.hidden_size*2, config.hidden_size2)
        self.fc2 = nn.Linear(config.hidden_size2, config.num_classes)

    def attention_net(self, x):
        #x: [B, L, H*2]
        scores = torch.tanh(torch.matmul(x, self.w_omega))
        #scores: [B, L, H*2]
        attention_weights = F.softmax(torch.matmul(scores, self.u_omega), dim=1)
        #attetion_weights: [B, L, 1]
        scored_outputs = scores * attention_weights
        #scored_outputs: [B, L, H*2]

        outputs = torch.sum(scored_outputs, 1)
        #outputs: [B, H*2]
        return outputs

    def forward(self, x):
        emb = self.embedding(x)
        H, _ = self.lstm(emb)
        attn_out = self.attention_net(H)
        out = F.relu(attn_out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


