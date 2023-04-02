from transformers import BartTokenizer
from transformers import BartModel

TRAIN_PATH = ""
TEST_PATH = ""
SAMPLE_NUM = 100 # max sample num is 347
TOKENIZER = BartTokenizer.from_pretrained("facebook/bart-base")
BART_MODEL = BartModel.from_pretrained("facebook/bart-base", output_hidden_states=True)