# coding: UTF-8
import torch
import os
import re
import sys
import pickle as pkl
import time
import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from torch.optim import optimizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.switch_backend('agg')

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

def build_vocab(dataset, tokenizer, max_size, min_freq):
    vocab_dic = {}
    sentences = dataset.iloc[:, 0].tolist()
    for s in sentences:
        for word in tokenizer(s):
            vocab_dic[word] = vocab_dic.get(word, 0)+1
    #根据词频率从大到小构成vocab_list
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1]>min_freq], key = lambda x:x[1], reverse=True)[:max_size]
    #vocab_dic = {word: idx}
    vocab_dic = { word_count[0]:idx for idx, word_count in enumerate(vocab_list)} 
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic)+1})
    return vocab_dic

def build_dataset(config, use_word):
    '''
    入参:
        config:      Config实例化
        use_word:    <FLAG>确定使用word_tokenizer还是char_tokenizer
    出参:
        vocab:       vocab_dict {word:idx}
    '''
    #load data
    data = pd.read_excel('./data.xls',usecols=[1,2])
    data.columns = ['text','label']
    #shuffle data
    data = shuffle(data)
    #clean text
    data.text = data.text.apply(cleanText)
    #encode label
    labelEncoder = LabelEncoder()
    labelEncoder.fit(data.label.tolist())
    data['label'] = labelEncoder.transform(data.label.tolist())

    #choose tokenizer
    if use_word:
        tokenizer = lambda x: x.split(' ') #use word
    else:
        tokenizer = lambda x:[y for y in x] #use char
    #construct vocab_dict
    if os.path.exists(config.vocab_path):                                                   #add config.vocab_path
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        os.makedirs('./data')
        vocab = build_vocab(data, tokenizer, max_size = MAX_VOCAB_SIZE, min_freq = 1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def _load_dataset(dataset, pad_size = 150):
        contents = []  #[ ([id,id...], 0/1)...]
        for idx, s in enumerate(dataset.iloc[:,0].tolist()):
            #获取label:
            label = dataset.iloc[idx, 1]
            #处理文本:
            words_line = [] 
            tokens = tokenizer(s)
            tokens_len = len(tokens)
            if pad_size:
                if tokens_len < pad_size: #如果tokens长度小于pad_size,补全
                    tokens.extend([PAD] * (pad_size - tokens_len))
                else: #如果过长，截断
                    tokens = tokens[:pad_size]
            #word to id
            for token in tokens:
                words_line.append(vocab.get(token, vocab.get(UNK)))
            
            contents.append((words_line, label))
        return contents

    total_datasets = _load_dataset(data, pad_size=150)
    #split into train/dev/test
    train_size, dev_size = int(0.8 * len(total_datasets)), int(0.1 * len(total_datasets))
    test_size = len(total_datasets) - train_size - dev_size
    train, dev, test = total_datasets[: train_size], total_datasets[train_size : train_size+dev_size], \
        total_datasets[train_size+dev_size: ]
    #保存一波处理好的数据集
    save_datasets(train, dev, test)
    return vocab

def save_datasets(train, dev, test):
    # if not os.path.exists('./datasets'):
    #     os.makedirs('./datasets')
    train_path = './datasets/train.pkl'
    dev_path = './datasets/dev.pkl'
    test_path = './datasets/test.pkl'
    with open(train_path,'wb') as train_pickle:
        pkl.dump(train, train_pickle)
    with open(dev_path,'wb') as dev_pickle:
        pkl.dump(dev, dev_pickle)
    with open(test_path,'wb') as test_pickle:
        pkl.dump(test, test_pickle)

def load_datasets(dataset_flag):
    path = './datasets/'+ dataset_flag + '.pkl'
    with open(path, 'rb') as f:
        dataset = pkl.load(f)
    return dataset


def cleanText(string):
    p = re.compile('^\s*[?.，．.-]?[(|（|\[]?[0-9|a-zA-Z]*[)|）|\]]?\s*[、|. ．]?')
    string = p.sub('',string)

    p_1 = re.compile('^\s*[①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩|●|-]?')
    string = p_1.sub('',string)
    
    p_2 = re.compile('^\s*[⑴|⑵|⑶|⑷|⑸|⑹|⑺|⑻|⑼|⑽|⑾|⑿|⒀|⒁|⒂|⒃|⒄|⒅|⒆|⒇]?')
    string = p_2.sub('',string)
    
    string = string.replace('\xa0','').replace('&lt','<').replace('&gt','>').lower()
    string = re.sub(r"[。;；, 、（）：，]", " ", string)
    return string.strip()

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def plot_train_loss(loss_collect, config):
    if not os.path.exists('./loss_fig'):
        os.makedirs('./loss_fig')
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(range(len(loss_collect)), loss_collect,'g.')
    ax.set_title('Loss For Model:{}'.format(config.model_name), fontsize=18)
    ax.set_xlabel('Num of Epochs', fontsize=18, fontstyle='italic')
    ax.set_ylabel('Loss(CE)', fontsize='x-large',fontstyle='oblique')

    # y_start, y_end = ax.get_ylim() 
    # ax.yaxis.set_ticks(np.arange(y_start, y_end, 0.1)) 

    plt.grid(True)

    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.savefig("./loss_fig/Loss For Model: {} {}.jpg".format(config.model_name, time_stamp))

class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        P = F.softmax(inputs, dim = 1)
        
        class_mask = torch.zeros_like(inputs)
        class_mask.scatter_(1, targets.view(-1,1), 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[targets.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #loss = -alpha * (1-probs)^gamma * log(probs)
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

# class ClinicalDataset(Dataset):
#     def __init__(self, datas, device):
#         super(ClinicalDataset, self).__init__()
#         self.datas = datas
#         self.device = device

#     def __len__(self):
#         return len(self.datas)
    
#     def __getitem__(self, index):
#         texts = self.datas[index][0]
#         label = self.datas[index][1]
#         texts = torch.LongTensor(texts).to(self.device)
#         label = torch.LongTensor(label).to(self.device)
#         return texts, label

# def build_iterator(dataset, config):
#     '''
#     input:
#         dataset: (string) 取值{'train','dev','test'}
#         config :  model的config
#     output:
#         iterator
#     '''
#     #加载datasets
#     dataset = load_datasets(dataset)
#     dataset = ClinicalDataset(dataset, config.device)
#     sampler = RandomSampler(dataset)
#     dataloader = DataLoader(dataset = dataset, sampler = sampler, batch_size = config.batch_size) #!add 'batch_size' to config
#     return dataloader

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        return x, y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    dataset = load_datasets(dataset)
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    vocab_dir = "./data/vocab.pkl"
    pretrain_dir = "./data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)