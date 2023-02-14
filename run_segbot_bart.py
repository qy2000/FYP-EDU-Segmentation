import re
from typing import List

from nltk.tokenize import word_tokenize
import numpy as np

import torch

from config import TOKENIZER
from import_data_bart import bart_tokenizer
from solver_bart import TrainSolver


def parse_input(inputstring: str):
    max_tokenizer_input_len = 128

    words = inputstring.split(" ")
    all_tokens = []
    all_masks = []
    all_boundaries = []

    while len(words) > max_tokenizer_input_len:
        cur_words = words[:128]
        words = words[128:]

        tokens, mask, boundaries = get_tokens_mask_boundaries(cur_words)

        all_tokens.append(np.asarray(tokens))
        all_masks.append(np.asarray(mask))
        all_boundaries.append(np.asarray(boundaries))

    tokens, mask, boundaries = get_tokens_mask_boundaries(words)

    all_tokens.append(np.asarray(tokens))
    all_masks.append(np.asarray(mask))
    all_boundaries.append(np.asarray(boundaries))

    all_tokens = np.asarray(all_tokens)
    all_masks = np.asarray(all_masks)
    all_boundaries = np.asarray(all_boundaries)

    return all_tokens, all_masks, all_boundaries


def get_tokens_mask_boundaries(cur_words: List):
    max_token_len = 256

    tokens, mask = bart_tokenizer(" ".join(cur_words))
    boundaries = [0 for _ in range(len(tokens) - 1)]
    boundaries.append(1)

    pad_tokens_count = max_token_len - len(tokens)
    pad = [1] * pad_tokens_count
    tokens.extend(pad)

    mask_remaining = [0] * pad_tokens_count
    mask.extend(mask_remaining)

    boundaries_remaining = [0] * pad_tokens_count
    boundaries.extend(boundaries_remaining)

    return tokens, mask, boundaries


def main_input_output(inputstring):
    X_in, X_mask, Y_in = parse_input(inputstring)
    print(X_in)

    mymodel = torch.load(r'model_epoch_5.torchsave', map_location=lambda storage, loc: storage)
    mymodel.use_cuda = False

    mymodel.eval()

    mysolver = TrainSolver(mymodel, train_x='', train_x_mask='', train_y='', dev_x='',
                            dev_x_mask='', dev_y='', save_path='',
                            batch_size=1, eval_size=1, epoch=10, lr=1e-2, lr_decay_epoch=1, weight_decay=1e-4,
                            use_cuda=False)

    all_visdata = []

    test_batch_ave_loss, test_pre, test_rec, test_f1, test_f1_token, visdata = mysolver.check_accuracy(X_in, X_mask, Y_in)
    print(visdata)

    start_b = visdata[3][0]
    end_b = visdata[2][0] + 1
    segments = []

    for i, END in enumerate(end_b):
        print(start_b[i], END)
        seg = TOKENIZER.decode(X_in[0][start_b[i]: END])
        print(seg)

    return segments


if __name__ == '__main__':

    #sent='Singapore recently announced that it is moving to a new Covid-19 innoculation strategy, with the focus on an individual’s vaccination being up-to-date, similar to how influenza jabs are administered seasonally. This comes as the country fights another wave of coronavirus infections, spurred by the emergence of the Omicron XBB sub-variant. With most people having received their primary vaccination series, as well as at least one booster, when should one take a second or third booster shot? The Straits Times asks the experts.'
    # sent = 'Sheraton and Pan Am said they are assured under the Soviet joint-venture law that they can repatriate profits from their hotel venture. They have been doing this for the past seventeen years and it has been successful.'
    sent = "The government is sharpening its newest weapon against white-collar defendants : the power to prevent them from paying their legal bills . And defense lawyers are warning that they won't stick around if they don't get paid . The issue has come to a boil in Newark , N.J. , where federal prosecutors have warned lawyers for Eddie Antar that if the founder and former chairman of Crazy Eddie Inc. is indicted , the government may move to seize the money that Mr. Antar is using to pay legal fees ."
    output_seg = main_input_output(sent)
    for ss in output_seg:
        print(ss)
