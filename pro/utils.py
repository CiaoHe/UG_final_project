# coding: UTF-8
import torch
import os
import re
from transformers import *
import sys
import pickle
import time
import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from torch.optim import optimizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')

PAD, CLS, SEP = 0, '[CLS]', '[SEP]'

def build_dataset(config):
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

    #initialize Tokenizer
    if config.BERT_USING in ['chinese_xlnet_mid_pytorch', 'chinese_xlnet_base_pytorch'] :
        tokenizer = XLNetTokenizer.from_pretrained(config.model_dir, do_lower_case = False)
        '''
        Attention! 使用xlnet时，由于模型特性抛弃了[mask]设置，因此在DataPrecess函数内需要进行特殊处理
        '''
    else:
        tokenizer = BertTokenizer.from_pretrained(config.model_dir, do_lower_case = False) 
        
    # 类初始化
    processor = DataPrecessForSingleSentence(tokenizer= tokenizer)
    # 产生输入数据
    seqs, seq_masks, seq_segments, labels = processor.get_input(dataset=data, max_seq_len=config.max_seq_len) 

    #split train/dev/test datasets
    train_size, dev_size = int(0.8 * len(seqs)), int(0.1 * len(seqs))
    test_size = len(seqs) - train_size - dev_size

    train, dev, test = ['', '', '', ''], ['', '', '', ''], ['', '', '', '']
    train[0], dev[0], test[0] = seqs[:train_size], seqs[train_size : train_size + dev_size], seqs[train_size + dev_size : ]
    train[1], dev[1], test[1] = seq_masks[:train_size], seq_masks[train_size : train_size + dev_size], seq_masks[train_size + dev_size : ]
    train[2], dev[2], test[2] = seq_segments[:train_size], seq_segments[train_size : train_size + dev_size], seq_segments[train_size + dev_size : ]
    train[3], dev[3], test[3] = labels[:train_size], labels[train_size : train_size + dev_size], labels[train_size + dev_size : ]

    # #save datasets:
    # if config.save_datasets == True: #!add 'save_datasets' to config:
    save_datasets(train, dev, test)

def save_datasets(train, dev, test):
    # if not os.path.exists('./datasets'):
    #     os.makedirs('./datasets')
    train_path = './datasets/train.pkl'
    dev_path = './datasets/dev.pkl'
    test_path = './datasets/test.pkl'
    with open(train_path,'wb') as train_pickle:
        pickle.dump(train, train_pickle)
    with open(dev_path,'wb') as dev_pickle:
        pickle.dump(dev, dev_pickle)
    with open(test_path,'wb') as test_pickle:
        pickle.dump(test, test_pickle)

def load_datasets(dataset_flag):
    path = './datasets/'+ dataset_flag + '.pkl'
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset
    
def build_iterator(dataset, config):
    '''
    input:
        dataset: (string) 取值{'train','dev','test'}
        config :  model的config
    output:
        iterator
    '''
    #加载datasets
    dataset = load_datasets(dataset)
    t_seqs = torch.tensor(dataset[0],dtype=torch.long)
    t_seq_masks = torch.tensor(dataset[1],dtype=torch.long)
    t_seq_segments = torch.tensor(dataset[2],dtype=torch.long)
    t_labels = torch.tensor(dataset[3],dtype=torch.long)
    
    data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(dataset = data, sampler = sampler, batch_size = config.batch_size) #!add 'batch_size' to config

    return dataloader


class DataPrecessForSingleSentence(object):
    """
    对文本进行处理
    """
    def __init__(self, tokenizer, max_workers=10):
        """
        tokenizer      :分词器
        dataset        :包含列名为'text'与'label'的pandas dataframe
        """
        self.tokenizer = tokenizer
        # 创建多线程池
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        # 获取文本与标签
        
    def get_input(self, dataset, max_seq_len=150):
        """
        通过多线程（因为notebook中多进程使用存在一些问题）的方式对输入文本进行分词、ID化、截断、填充等流程得到最终的可用于模型输入的序列。
        
        入参:
            dataset     : pandas的dataframe格式，包含两列，第一列为文本，第二列为标签。
            max_seq_len : 目标序列长度，该值需要预先对文本长度进行分别得到，可以设置为小于等于512（BERT的最长文本序列长度为512）的整数。
        
        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，因为是单句，所以取值都为0。
            labels      : 标签取值为{0~len(labels)-1}。
        
            
        """
        sentences = dataset.iloc[:, 0].tolist()
        labels = dataset.iloc[:, 1].tolist()
        # 切词
        tokens_seq = list(
            self.pool.map(self.tokenizer.tokenize, sentences))
        # 获取定长序列及其mask
        result = list(
            self.pool.map(self.trunate_and_pad, tokens_seq,
                          [max_seq_len] * len(tokens_seq)))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return seqs, seq_masks, seq_segments, labels
    
    def trunate_and_pad(self, seq, max_seq_len):
        """
        1. 因为本类处理的是单句序列，按照BERT中的序列处理方式，需要在输入序列头尾分别拼接特殊字符'CLS'与'SEP'，
           因此不包含两个特殊字符的序列长度应该小于等于max_seq_len-2，如果序列长度大于该值需要那么进行截断。
        2. 对输入的序列 最终形成['CLS',seq,'SEP']的序列，该序列的长度如果小于max_seq_len，那么使用0进行填充。
        
        入参: 
            seq         : 输入序列，在本处其为单个句子。
            max_seq_len : 拼接'CLS'与'SEP'这两个特殊字符后的序列长度
        
        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，因为是单句，所以取值都为0。
           
        """
        # 对超长序列进行截断
        if len(seq) > (max_seq_len - 2):
            seq = seq[0:(max_seq_len - 2)]
        # 分别在首尾拼接特殊符号
        seq = [CLS] + seq + [SEP]
        # ID化
        seq = self.tokenizer.convert_tokens_to_ids(seq)
        # 根据max_seq_len与seq的长度产生填充序列
        padding = [PAD] * (max_seq_len - len(seq))
        # 创建seq_mask
        seq_mask = [1] * len(seq) + padding
        # 创建seq_segment
        seq_segment = [0] * len(seq) + padding
        # 对seq拼接填充序列
        seq += padding
        assert len(seq) == max_seq_len
        assert len(seq_mask) == max_seq_len
        assert len(seq_segment) == max_seq_len
        return seq, seq_mask, seq_segment

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

