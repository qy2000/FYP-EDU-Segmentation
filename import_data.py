import os
from typing import List
import string
import re
from transformers import BartTokenizer
import numpy as np

# TODO: change current code such that BART tokenizer tokenizes which discourse unit
#  based on the separators and add boundaries accordingly
#  convert all list to nd array

PATH = "C:/Users/qingy/Downloads/FYP/RST_SEGMENTATION_DATA/RST_SEGMENTATION_DATA/SEN_WITH_EDU/TRAINING/"
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", add_prefix_space=True)

# pattern = fr'[{re.escape(string.punctuation)}]'


def bart_tokenizer(text: str) -> List[int]:
    tokens = tokenizer.encode(text)
    # print(tokens)
    dec = tokenizer.decode(tokens)
    # print(dec)
    # remove these tokens
    # [CLS]: 0
    # [SEP]: 2
    # < pad >: 1
    tokens = tokens[1:-1]
    # print(tokens)
    dec = tokenizer.decode(tokens)
    # print(dec)
    return tokens


def read_data(path: str):
    all_files = os.listdir(path)
    all_words = []
    all_tokens = []
    all_boundaries = []
    for file in all_files[:100]:
        with open(PATH + file, 'r') as f:
            words, tokens, boundaries = parse_file_text(f.read())
            all_words.append(words)

            while len(tokens) > 1000:
                all_tokens.append(tokens[:1000])
                all_boundaries.append(boundaries[:1000])
                tokens = tokens[1000:]
                boundaries = boundaries[1000:]

        all_tokens.append(tokens)
        all_boundaries.append(boundaries)

    return all_words, all_tokens, all_boundaries


def parse_file_text(file_text: str):
    words = file_text.split("EDU_BREAK")
    tokens = []
    boundaries = []
    for word in words:
        token = bart_tokenizer(word)
        token = np.asarray(token)
        tokens.extend(token)
        boundary = [0 for _ in range(len(token) - 1)]
        boundary.append(1)
        boundary = np.asarray(boundary)
        boundaries.extend(boundary)
        # print(len(tokens), len(boundaries))

    tokens = np.asarray(tokens)
    boundaries = np.asarray(boundaries)


    return words, tokens, boundaries




read_data(PATH)
