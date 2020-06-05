# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import precision_recall_curve,classification_report
import time
from tqdm import tqdm_notebook, trange, tqdm
from utils_old import plot_train_loss, FocalLoss

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
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
    # init_network(model)

    max_grad_norm = 1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    loss_collect = []
    total_batch = 0  #记录进行了多少轮batch
    for epoch in trange(config.num_epochs, desc='Epoch'):
        for step, (trains, labels) in enumerate(tqdm(train_iter, desc='Iteration')):
            # Forward Pass
            outputs = model(trains)
            # Backward and optimizer
            optimizer.zero_grad() 
            if config.use_FocalLoss:
                FL_loss = FocalLoss(config.num_classes)
                loss = FL_loss(outputs, labels)
            else:
                loss = F.cross_entropy(outputs, labels)
            # Backward Propogation
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
    print(dev_report)
    print('dev_acc:', dev_acc, 'dev_loss:', dev_loss)

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
        for (batch_data, labels) in data_iter:
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

    #print(predict_all)
    acc = metrics.accuracy_score(labels_all, predict_all)
    report = metrics.classification_report(labels_all, predict_all, digits = 4)
    if test_flag:
        report = metrics.classification_report(labels_all, predict_all, digits = 4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, (loss_total / len(data_iter)).cpu().numpy(), report, confusion
    return acc, (loss_total / len(data_iter)).cpu().numpy(), report

