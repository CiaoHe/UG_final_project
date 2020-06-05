#!/bin/sh

python run.py --model bert_att --PTM chinese_rbtl3_pytorch \
--num_epochs 10 --batch_size 64 --focal_loss Y \
--DISCR Y --STLR Y

# python run.py --model xlnet_CNN --PTM chinese_xlnet_base_pytorch \
# --num_epochs 5 --batch_size 32 --focal_loss Y \
# --DISCR Y --STLR Y

# python run.py --model xlnet_att --PTM chinese_xlnet_base_pytorch \
# --num_epochs 5 --batch_size 32 --focal_loss Y \
# --DISCR Y --STLR Y