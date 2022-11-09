from import_data_bart import read_data, PATH

import numpy as np

import os
import torch
from solver_bart import TrainSolver
from model_bart_v2 import PointerNetworks
all_tokens, all_masks, all_boundaries = read_data()
print("len all tokens:", len(all_tokens))
idx = int(len(all_tokens)*0.7)
train_x = all_tokens[:idx]
train_x_mask = all_masks[:idx]
train_y = all_boundaries[:idx]
test_x = all_tokens[idx:]
test_x_mask = all_masks[idx:]
test_y = all_boundaries[idx:]
SAVE_PATH = "C:/Users/qingy/Downloads/FYP/RunSegBot/TrainResults100BART_samelen1"

my_model = PointerNetworks(encoder_type='BART', decoder_type='GRU', rnn_layers=6,
                           dropout_prob=0.5, use_cuda=False)

my_solver = TrainSolver(my_model, train_x=train_x, train_x_mask=train_x_mask, train_y=train_y, dev_x=test_x, dev_x_mask=test_x_mask, dev_y=test_y, save_path=SAVE_PATH,
                        batch_size=1, eval_size=1, epoch=10, lr=1e-2, lr_decay_epoch=1, weight_decay=1e-4,
                        use_cuda=False)

my_solver.train()
