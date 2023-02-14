import torch
from transformers import AutoTokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
PATH = "C:/Users/qingy/Downloads/FYP/RST_SEGMENTATION_DATA/RST_SEGMENTATION_DATA/SEN_WITH_EDU/TRAINING/"
SAMPLE_NUM = 100
SAVE_PATH="C:/Users/qingy/Downloads/FYP/RunSegBot/BERTTokenClassification/Results"