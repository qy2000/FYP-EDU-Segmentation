from import_data import read_data, PATH

import numpy as np

import os
import torch
from solver import TrainSolver
from model_bart_v1 import PointerNetworks
# from model import PointerNetworks
all_text, all_tokens, all_boundaries = read_data(PATH)
print(len(all_tokens))
idx = int(len(all_tokens)*0.7)
train_x = np.array(all_tokens[:idx], dtype=object)
train_y = np.array(all_boundaries[:idx], dtype=object)
test_x = np.array(all_tokens[idx:], dtype=object)
test_y = np.array(all_boundaries[idx:], dtype=object)
SAVE_PATH = "C:/Users/qingy/Downloads/FYP/RunSegBot/TrainResults100BART_1000"
# os.rmdir(SAVE_PATH)

# my_model = PointerNetworks(voca_size=80, voc_embeddings=np.ndarray(shape=(80, 300), dtype=float), word_dim=300,
#                            hidden_dim=10, is_bi_encoder_rnn=True, rnn_type='GRU', rnn_layers=3,
#                            dropout_prob=0.5, use_cuda=False, finedtuning=True, isbanor=True)

my_model = PointerNetworks(voca_size=2, voc_embeddings=np.ndarray(shape=(2, 300), dtype=float), word_dim=300,
                           hidden_dim=10, is_bi_encoder_rnn=True, encoder_type='BART', decoder_type='GRU', rnn_layers=6,
                           dropout_prob=0.5, use_cuda=False, finedtuning=True, isbanor=True)

my_solver = TrainSolver(my_model, train_x=train_x, train_y=train_y, dev_x=test_x, dev_y=test_y, save_path=SAVE_PATH,
                        batch_size=1, eval_size=1, epoch=10, lr=1e-2, lr_decay_epoch=1, weight_decay=1e-4,
                        use_cuda=False)

my_solver.train()
