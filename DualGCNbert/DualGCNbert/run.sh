#!/bin/bash


# * DualGCNbert
CUDA_VISIBLE_DEVICES=0 python3 ../'Active_Learning_Pytorch.py' --model_name dualgcnbert --dataset twitter --seed 1000 --bert_lr 2e-5 --num_epoch 5 --hidden_dim 768 --max_length 150 --cuda 0 --losstype doubleloss --alpha 0.5 --beta 0.9 --parseadj --num_layers 2

