# coding: UTF-8
import time
import torch
import numpy as np
import pickle as pkl
import os 
import argparse
from train_eval_old import train, init_network
from importlib import import_module
from utils_old import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', 
                    default = 'TextCNN',
                    type=str, required=True, 
                    help='choose a model: TextCNN/TextRNN/DPCNN')
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
parser.add_argument('--save_loss',
                    default = 'N',
                    type = str,
                    help='Whether to save the figure of training_loss (Y/N)')
parser.add_argument('--embedding',
                    default='pre_trained', 
                    type = str, 
                    help='random or pre_trained')
parser.add_argument('--word', 
                    default=False, 
                    type=bool, 
                    help='True for word, False for char')


args = parser.parse_args()

if __name__ == '__main__':
    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz' #default
    if args.embedding == 'random':
        embedding = 'random'

    #导入选择使用的模型
    model_name = args.model
    model = import_module('.' + model_name, 'models')
    #确实是否要保留training loss图像
    if args.save_loss == 'Y':
        save_loss_flag = True
    else:
        save_loss_flag = False

    #确定num_epochs, batch_size
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    config = model.Config(embedding, num_epochs, batch_size)

    #确定是否用focal_loss
    if args.focal_loss == 'Y':
        if not hasattr(config,'use_FocalLoss'):
            setattr(config,'use_FocalLoss',True)
    else:
        setattr(config,'use_FocalLoss',False)

    # random initialize
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    if not os.path.exists('./datasets'):
        os.makedirs('./datasets')
        vocab = build_dataset(config, args.word)
    elif os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))

    config.n_vocab = len(vocab)
    train_iter = build_iterator('train', config)
    dev_iter = build_iterator('dev', config)
    test_iter = build_iterator('test', config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # for data,label in test_iter:
    #     print(data.shape)
    #     break


    # train
    model = model.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter, save_loss = save_loss_flag)
