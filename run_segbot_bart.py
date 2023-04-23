
import warnings
import re
from typing import List

import numpy as np

import torch

from config import TEST_PATH, TOKENIZER
from import_data_bart import read_data
from solver_bart import TrainSolver

import os
import time


warnings.filterwarnings("ignore")

start_time = time.time()
def bart_tokenizer(text: str) -> List[int]:
    '''
    :param text:
    :return:
    Add special tokens to the start and end of each sentence
    Pad & truncate all sentences to a single constant length.
    Explicitly differentiate real tokens from padding tokens with the “attention mask”.

    '''
    tokens = TOKENIZER.encode_plus(
        text,                      # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        return_attention_mask=True,   # Construct attn. masks.
    )

    dec = TOKENIZER.decode(tokens['input_ids'])
    # print(dec)

    # remove sep tokens
    return tokens['input_ids'][1:-1], tokens['attention_mask'][1:-1]


def parse_input(inputstring: str):
    '''
    Split sentences by the full stop and form input sequences with a token length of 128
    '''
    max_token_len = 128

    sentences = inputstring.split(" . ")
    all_tokens = []
    all_masks = []
    all_boundaries = []

    cur_tokens = []
    cur_mask = []
    cur_boundaries = []
    cur = 0
    i = 0

    while i < len(sentences):
        if i != len(sentences) - 1:
            cur_sent = sentences[i] + " . "
        else:
            cur_sent = sentences[i]

        tokens, mask = bart_tokenizer(cur_sent)
        if len(tokens) + cur <= max_token_len:
            cur += len(tokens)
            cur_tokens.extend(tokens)
            cur_mask.extend(mask)

            boundaries = [0 for _ in range(len(tokens) - 1)]
            boundaries.append(1)
            cur_boundaries.extend(boundaries)
            i += 1

        else:
            pad_tokens_count = max_token_len - cur
            pad = [1] * pad_tokens_count
            cur_tokens.extend(pad)

            mask_remaining = [0] * pad_tokens_count
            cur_mask.extend(mask_remaining)

            boundaries_remaining = [0] * pad_tokens_count
            cur_boundaries.extend(boundaries_remaining)

            all_tokens.append(np.asarray(cur_tokens))
            all_masks.append(np.asarray(cur_mask))
            all_boundaries.append(np.asarray(cur_boundaries))

            cur_tokens = []
            cur_mask = []
            cur_boundaries = []
            cur = 0

    if (cur_tokens != []):
        pad_tokens_count = max_token_len - len(cur_tokens)
        pad = [1] * pad_tokens_count
        cur_tokens.extend(pad)

        mask_remaining = [0] * pad_tokens_count
        cur_mask.extend(mask_remaining)

        boundaries_remaining = [0] * (max_token_len - len(cur_boundaries))
        cur_boundaries.extend(boundaries_remaining)

        all_tokens.append(np.asarray(cur_tokens))
        all_masks.append(np.asarray(cur_mask))
        all_boundaries.append(np.asarray(cur_boundaries))

    return all_tokens, all_masks, all_boundaries


def main_input_output(inputstring):
    '''
    Load trained model and run the inference on each input sequence
    '''
    X_in, X_mask, Y_in = parse_input(inputstring)
    segments = []

    mymodel = torch.load(r'model_segbot_bart_final.torchsave',
                         map_location=lambda storage, loc: storage)
    mymodel.use_cuda = False

    mymodel.eval()

    mysolver = TrainSolver(mymodel, train_x='', train_x_mask='', train_y='', dev_x='',
                           dev_x_mask='', dev_y='', save_path='',
                           batch_size=1, eval_size=1, epoch=10, lr=0.00015, lr_decay_epoch=1, weight_decay=0.0002,
                           use_cuda=False)

    for i in range(len(X_in)):
        start_time = time.time()
        cur_X_in = np.asarray([X_in[i]])
        cur_X_mask = np.asarray([X_mask[i]])
        cur_Y_in = np.asarray([Y_in[i]])

        all_visdata = []

        test_batch_ave_loss, test_pre, test_rec, test_f1, visdata = mysolver.check_accuracy(
            cur_X_in, cur_X_mask, cur_Y_in)

        start_b = visdata[3][0]
        end_b = visdata[2][0] + 1

        for j, END in enumerate(end_b):
            print(start_b[j], end_b[j])
            seg = TOKENIZER.decode(X_in[i][start_b[j]: END])
            print(seg)

        print("--- %s seconds ---" % (time.time() - start_time))

    return segments


if __name__ == '__main__':
    #sent="In ASEAN, there are currently government initiatives to encourage renewable energy, with Singapore predicting that hydrogen could supply up to half of the power needs in Singapore by 2050 and Thailand with a Hydrogen goal of 10 Kilotons of oil equivalent in total by 2036."
    #sent="Furthermore, the current advancements in technology for hydrogen energy is able to reduce costs in terms of production and storage of hydrogen energy. As the technology continues to improve, it is expected to further lower the cost of production, achieving economies of scale."
    #sent='Singapore recently announced that it is moving to a new Covid-19 innoculation strategy, with the focus on an individual’s vaccination being up-to-date, similar to how influenza jabs are administered seasonally. This comes as the country fights another wave of coronavirus infections, spurred by the emergence of the Omicron XBB sub-variant. '
    #     sent="Aerial warfare has been around for much longer than modern aircraft have. More than 1,000 years ago, armies in China used incendiary kites known as fire crows to rain fire and debris upon their enemies. Since then, everything from kites to hot air balloons and airplanes have been used to inflict damage from above."
    # print("----------- EDU Segmentation with Segbot with BART model: ----------")
    # sent = input("Enter text for EDU segmentation: \n")
    # sent = sent.replace(", ",  " , ").replace(". ",  " . ").replace(
    #     "; ",  " ; ")
    # if sent[-1] == ".":
    #     sent = sent[:-1] + " ."
    # print("\n")
    # print("---------- Start of EDU segmentation ----------")
    # output_seg = main_input_output(sent)
    # print("---------- End of EDU segmentation ----------\n")

    '''
    Get inference time by each new line of input
    '''
    # all_files = os.listdir(TEST_PATH)
    # total = 0
    # for file in all_files:
    #     with open(TEST_PATH + file, 'r') as f:
    #         file_text = f.read()
    #         sentences = file_text.split("\n")
    #         total += len(sentences)
    #         for sent in sentences:
    #             print(sent)
    #             try:
    #                 new_sent = sent.replace(" EDU_BREAK", "")
    #                 output_seg =  main_input_output(new_sent)
    #             except Exception as e:
    #                 print(e)
    # print(total)
    # print("Total inference time")
    # print("--- %s seconds ---" % (time.time() - start_time))



