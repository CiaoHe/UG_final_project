# coding: UTF-8
import time
import torch
import numpy as np
import os 
import argparse
from train_eval import train, init_network
from importlib import import_module
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', 
                    default = 'bert',
                    type=str, required=True, 
                    help='choose a model: bert/bert_CNN/bert_DPCNN/bert_lstm')
parser.add_argument('--PTM', 
                    default = 'chinese_rbtl3_pytorch',
                    type=str,  
                    help='choose pre-trained models: chinese_rbtl3_pytorch/chinese_wwm_ext_pytorch/chinese_wwm_pytorch/xlnet')
parser.add_argument('--num_epochs',
                    default = 2,
                    type = int,
                    help='Total number of training epochs to perform')
parser.add_argument('--batch_size',
                    default = 64,
                    type = int,
                    help = 'Total batch size')
parser.add_argument('--focal_loss',
                    default = 'N',
                    type = str,
                    help='Whether to use focal_loss (Y/N)')
parser.add_argument('--DISCR',
                    default = 'N',
                    type = str,
                    help='default N')
parser.add_argument('--STLR',
                    default = 'N',
                    type = str,
                    help='default N')
parser.add_argument('--save_loss',
                    default = 'N',
                    type = str,
                    help='Whether to save the figure of training_loss (Y/N), default N')


args = parser.parse_args()

if __name__ == '__main__':
    #导入选择使用的模型
    model_name = args.model
    model_PTM = args.PTM
    model = import_module('.' + model_name, 'models')
    #确定num_epochs, batch_size
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    #确实是否要保留training loss图像
    if args.save_loss == 'Y':
        save_loss_flag = True
    else:
        save_loss_flag = False

    config = model.Config(num_epochs, batch_size, model_PTM)
    #确定是否用focal_loss
    if args.focal_loss == 'Y':
        if not hasattr(config,'use_FocalLoss'):
            setattr(config,'use_FocalLoss',True)
    else:
        setattr(config,'use_FocalLoss',False)
    #确定是否用DISCR
    if args.DISCR == 'Y':
        if not hasattr(config,'DISCR'):
            setattr(config,'DISCR',True)
    else:
        setattr(config,'DISCR',False)
    #确定是否用STLR
    if args.STLR == 'Y':
        if not hasattr(config,'STLR'):
            setattr(config,'STLR',True)
    else:
        setattr(config,'STLR',False)

    # random initialize
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    if not os.path.exists('./datasets'):
        os.makedirs('./datasets')
        build_dataset(config)
    train_iter = build_iterator('train', config)
    dev_iter = build_iterator('dev', config)
    test_iter = build_iterator('test', config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = model.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter, save_loss = save_loss_flag)
