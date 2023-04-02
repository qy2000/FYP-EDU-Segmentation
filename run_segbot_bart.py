
from typing import List

import numpy as np

import torch

from config import TEST_PATH, TOKENIZER
from solver_bart import TrainSolver

import os
import time
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
    max_tokenizer_input_len = 256

    words = inputstring.split(" ")
    all_tokens = []
    all_masks = []
    all_boundaries = []

    while len(words) > max_tokenizer_input_len:
        cur_words = words[:256]
        words = words[256:]

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

    mymodel = torch.load(r'model_epoch_0_optuna_37.torchsave', map_location=lambda storage, loc: storage)
    mymodel.use_cuda = False

    mymodel.eval()

    mysolver = TrainSolver(mymodel, train_x='', train_x_mask='', train_y='', dev_x='',
                            dev_x_mask='', dev_y='', save_path='',
                            batch_size=1, eval_size=1, epoch=10, lr=0.00016, lr_decay_epoch=1, weight_decay=0.0002,
                            use_cuda=False)

    all_visdata = []

    test_batch_ave_loss, test_pre, test_rec, test_f1, visdata = mysolver.check_accuracy(X_in, X_mask, Y_in)

    start_b = visdata[3][0]
    end_b = visdata[2][0] + 1
    segments = []



    for i, END in enumerate(end_b):
        print(start_b[i], end_b[i])
        seg = TOKENIZER.decode(X_in[0][start_b[i]: END])
        seg_final = seg.replace("<pad>", "")
        print(seg_final)

    return segments


if __name__ == '__main__':
    sent='Singapore recently announced that it is moving to a new Covid-19 innoculation strategy, with the focus on an individual’s vaccination being up-to-date, similar to how influenza jabs are administered seasonally. This comes as the country fights another wave of coronavirus infections, spurred by the emergence of the Omicron XBB sub-variant. '
    #sent="Aerial warfare has been around for much longer than modern aircraft have. More than 1,000 years ago, armies in China used incendiary kites known as fire crows to rain fire and debris upon their enemies. Since then, everything from kites to hot air balloons and airplanes have been used to inflict damage from above."
    sent = sent.replace(',', ' ,').replace('.', ' .')
    all_files = os.listdir(TEST_PATH)
    total = 0
    for file in all_files:
        with open(TEST_PATH + file, 'r') as f:
            file_text = f.read()
            sentences = file_text.split("\n")
            total += len(sentences)
            for sent in sentences:
                try:
                    print(sent)
                    new_sent = sent.replace(" EDU_BREAK", "")
                    output_seg =  main_input_output(new_sent)
                    for ss in output_seg:
                        print(ss)
                except Exception as e:
                    print(e)
    print(total)
    print("Total inference time")
    print("--- %s seconds ---" % (time.time() - start_time))
    output_seg = main_input_output(sent)
