# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import precision_recall_curve,classification_report
import time
from transformers import *
from tqdm import tqdm_notebook, trange, tqdm
from utils import plot_train_loss, FocalLoss
import logging
import pickle

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def train(config, model, train_iter, dev_iter, test_iter, save_loss = False):
    model.train()
    init_network(model)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    
    lr = config.learning_rate
    max_grad_norm = 1.0
    num_training_steps = 1000
    num_warmup_steps = 100
    warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1

    def differential_params(model, init_lr, beta_decay=0.9,):
        try:
            num_layers = len(model.bert.encoder.layer)
        except AttributeError:
            return model.parameters()
        #filter out layer_params to get the other params
        layer_params = []
        for layer_id in range(num_layers):
            layer_params += list(map(id, model.bert.encoder.layer[layer_id].parameters()))
        base_params = filter(lambda p: id(p) not in layer_params, model.parameters())
        #differential bert layer's lr
        layer_params_lr = []
        for layer_id in range(num_layers-1,-1,-1):
            layer_params_lr_dict = {}
            layer_params_lr_dict['params'] = model.bert.encoder.layer[layer_id].parameters()
            layer_params_lr_dict['lr'] = round(init_lr*(beta_decay)**layer_id,9)
            layer_params_lr.append(layer_params_lr_dict)
        #return the new joint model parameters
        model_parameters = [{'params':base_params}] + layer_params_lr

        model.parameters()
        return model_parameters

    #set the torch.optimizer according whether using DISCR
    if config.DISCR:
        optimizer = AdamW(differential_params(model, init_lr = lr), lr=lr, correct_bias=False)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)
    #set the scheduler according whether using STLR, default just using warming_up
    if config.STLR:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler
    else:
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)

    loss_collect = []
    total_batch = 0  #记录进行了多少轮batch
    for epoch in trange(config.num_epochs, desc='Epoch'):
        for step, batch_data in enumerate(tqdm(train_iter, desc='Iteration')):
            batch_data = tuple(t.to(config.device) for t in batch_data)
            labels = batch_data[-1]
            # Forward Pass
            outputs = model(batch_data)
            # Backward and optimizer
            optimizer.zero_grad() 
            if config.use_FocalLoss:
                FL_loss = FocalLoss(config.num_classes)
                loss = FL_loss(outputs, labels)
            else:
                loss = F.cross_entropy(outputs, labels)
            loss.backward()
            loss_collect.append(loss.item())
            print("\r%f" % loss, end='')
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

    #保存loss图像：
    if save_loss:
        plot_train_loss(loss_collect, config)

    #在dev集上做验证
    dev_acc, dev_loss, dev_report = evaluate(config, model, dev_iter)
    # print(dev_report)
    # print('dev_acc:', dev_acc, 'dev_loss:', dev_loss)

    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("log.txt",'a')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
     
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
     
    logger.addHandler(handler)
    logger.addHandler(console)
     
    logger.info('USING MODEL: %s, Using PTM: %s' % (config.model_name, config.BERT_USING))
    logger.info('Batch_Size: %d, Using FL: %s, Using DISCR: %s, Using STLR: %s' % (config.batch_size, 
        config.use_FocalLoss, config.DISCR, config.STLR))
    #print(dev_report)
    with open('log.txt','a+') as f:
        print(dev_report, file = f)
    logger.info('dev_acc: %s  dev_loss: %s' %(dev_acc, dev_loss))
    logger.info('-----------------------------------------------------------\n')

    #在test集求结果
    # test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test_flag = True)
    # print(test_report)
    # print('test_acc:', test_acc, 'test_loss:', test_loss)
    #print('test_confusion\n', test_confusion)


def evaluate(config, model, data_iter, test_flag = False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch_data in data_iter:
            batch_data = tuple(t.to(config.device) for t in batch_data)
            labels = batch_data[-1]
            # Forward Pass
            outputs = model(batch_data)
            # Compute Loss
            if config.use_FocalLoss:
                FL_loss = FocalLoss(config.num_classes)
                loss = FL_loss(outputs, labels)
            else:
                loss = F.cross_entropy(outputs, labels)
            loss_total += loss

            # Append into Final Predics&Labels
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, dim = 1)[1].cpu().numpy()  #torch.max() [0]:返回值 [1]:返回索引
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    #print clf report
    with open('./num2label_dic.pkl','rb') as f:
        num2label_dic = pickle.load(f)
        num2label = [num2label_dic[i] for i in set(labels_all)]
    report = metrics.classification_report(labels_all, predict_all, target_names=num2label, digits = 4)
    if test_flag:
        report = metrics.classification_report(labels_all, predict_all, target_names=num2label, digits = 4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, (loss_total / len(data_iter)).cpu().numpy(), report, confusion
    return acc, (loss_total / len(data_iter)).cpu().numpy(), report
