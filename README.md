# UG_final_project

## 简介
本项目为哈工大（深圳）16级计算机毕设《中文临床试验文本主题分类》。主要包含old(基于预训练词向量的文本分类模型）和pro（基于微调深度PTMs的文本分类模型）两部分。使用框架：pytorch1.1.0，Transformers工具包（HuggingFace）。深度PTMs来源于哈工大讯飞联合实验室发布的BERT/RoBERTa/XLNet等框架。

## 框架
* old
  * datasets(save train/val/test.pkl)
  * models(各类网络模型参数）
  * run_old.py(main)
  * train_eval_old.py(训练和验证函数）
  * utils_old.py(数据预处理、FL实现等工具函数）
  * pre-trained embeddings (保存预训练词向量）
  
* pro
  * --基本同上--
  * script.sh (自动测试脚本）
  * loss_fig （用于保存训练损失函数图像）
  * models（保存各类网络模型参数 & 保存各类PTMs的网络参数/分词模型）

## 使用说明
训练/验证
``` shell
cd pro
python run.py -h

usage: run.py [-h] --model MODEL [--PTM PTM] [--num_epochs NUM_EPOCHS]
              [--batch_size BATCH_SIZE] [--focal_loss FOCAL_LOSS]
              [--DISCR DISCR] [--STLR STLR] [--save_loss SAVE_LOSS]

Chinese Text Classification
optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         choose a model: bert/bert_CNN/bert_DPCNN/bert_lstm
  --PTM PTM             choose pre-trained models: chinese_rbtl3_pytorch/chine
                        se_wwm_ext_pytorch/chinese_wwm_pytorch/xlnet
  --num_epochs NUM_EPOCHS
                        Total number of training epochs to perform
  --batch_size BATCH_SIZE
                        Total batch size
  --focal_loss FOCAL_LOSS
                        Whether to use focal_loss (Y/N)
  --DISCR DISCR         default N
  --STLR STLR           default N
  --save_loss SAVE_LOSS
                        Whether to save the figure of training_loss (Y/N),
                        default N
```
OR
``` shell
cd pro
script.sh
```

