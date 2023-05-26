from typing import List

import warnings
warnings.filterwarnings('ignore')

import torch
import transformers
import numpy as np

from config import TOKENIZER


def bert_tokenizer(text: str) -> List[int]:
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
    Split sentences by the full stop and form input sequences with a token length of 256
    '''
    max_token_len = 256

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

        tokens, mask = bert_tokenizer(cur_sent)
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
    x, x_mask, y = parse_input(inputstring)
    model = transformers.BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=2)

    # Load the state dictionary
    state_dict = torch.load(r'BERT_token_classification_cased1.pth',
                         map_location=torch.device('cpu'))

    # Remove the "module." prefix from the state keys
    new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}


    model.load_state_dict(new_state_dict)
    model = model.to(torch.device('cpu'))
    model.eval()

    x = torch.tensor(x, dtype=torch.int64).to(torch.device('cpu'), dtype=torch.int64)
    x_mask = torch.tensor(x_mask, dtype=torch.int64).to(torch.device('cpu'), dtype=torch.int64)

    with torch.no_grad():
        output = model(x, token_type_ids=None, attention_mask=x_mask)
        predictions = np.argmax(output[0].detach().numpy(), axis=2)
        boundaries = [np.where(arr == 1)[0] for arr in predictions]

    for i in range(len(boundaries)):
        if (len(boundaries[i]) == 0):
            print("No boundaries found")
            seg = TOKENIZER.decode(x[i])
            print(seg)
        else:
            start = 0
            for boundary in boundaries[i]:
                if (start == 0 or start != boundary):
                    print(start, boundary)
                    seg = TOKENIZER.decode(x[i][start:boundary+1])
                    start = boundary + 1
                    print(seg)
                else:
                    continue


if __name__ == '__main__':
    #sent="In ASEAN, there are currently government initiatives to encourage renewable energy, with Singapore predicting that hydrogen cohiuld supply up to half of the power needs in Singapore by 2050 and Thailand with a Hydrogen goal of 10 Kilotons of oil equivalent in total by 2036."
    #sent="Furthermore, the current advancements in technology for hydrogen energy is able to reduce costs in terms of production and storage of hydrogen energy. As the technology continues to improve, it is expected to further lower the cost of production, achieving economies of scale."
    #sent='Singapore recently announced that it is moving to a new Covid-19 innoculation strategy, with the focus on an individual’s vaccination being up-to-date, similar to how influenza jabs are administered seasonally. This comes as the country fights another wave of coronavirus infections, spurred by the emergence of the Omicron XBB sub-variant. '
    #     sent="Aerial warfare has been around for much longer than modern aircraft have. More than 1,000 years ago, armies in China used incendiary kites known as fire crows to rain fire and debris upon their enemies. Since then, everything from kites to hot air balloons and airplanes have been used to inflict damage from above."
    print("----------- EDU Segmentation with BERT token classification model: ----------")
    sent = input("Enter text for EDU segmentation: \n")
    sent = sent.replace(", ",  " , ").replace(". ",  " . ").replace(
        "; ",  " ; ")
    if sent[-1] == ".":
        sent = sent[:-1] + " ."
    print("\n")
    print("---------- Start of EDU segmentation ----------")
    output_seg = main_input_output(sent)
    print("---------- End of EDU segmentation ----------\n")