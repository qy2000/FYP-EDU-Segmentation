import torch
from transformers import AutoTokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-cased")
PATH = ""
SAMPLE_NUM = 100
SAVE_PATH=""