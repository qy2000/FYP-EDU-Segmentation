from import_data_bart import read_data
from config import TRAIN_PATH, TEST_PATH

from solver_bart import TrainSolver
from model_bart_v2 import PointerNetworks
train_tokens, train_masks, train_boundaries = read_data(TRAIN_PATH)
print("len train tokens:", len(train_tokens))
train_x = train_tokens
train_x_mask = train_masks
train_y = train_boundaries
test_tokens, test_masks, test_boundaries = read_data(TEST_PATH)
print("len test tokens:", len(test_tokens))
test_x = test_tokens
test_x_mask = test_masks
test_y = test_boundaries
SAVE_PATH = "C:/Users/qingy/Downloads/FYP/RunSegBot/TrainResults100BART_samelen3"

my_model = PointerNetworks(encoder_type='BART', decoder_type='GRU', rnn_layers=6,
                           dropout_prob=0.5, use_cuda=False)

my_solver = TrainSolver(my_model, train_x=train_x, train_x_mask=train_x_mask, train_y=train_y, dev_x=test_x, dev_x_mask=test_x_mask, dev_y=test_y, save_path=SAVE_PATH,
                        batch_size=1, eval_size=1, epoch=10, lr=1e-2, lr_decay_epoch=1, weight_decay=1e-4,
                        use_cuda=False)

my_solver.train()
