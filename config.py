from transformers import BartTokenizer
from transformers import BartModel

TRAIN_PATH = "C:/Users/qingy/Downloads/FYP/RST_SEGMENTATION_DATA/RST_SEGMENTATION_DATA/SEN_WITH_EDU/TRAINING/"
TEST_PATH = "C:/Users/qingy/Downloads/FYP/RST_SEGMENTATION_DATA/RST_SEGMENTATION_DATA/SEN_WITH_EDU/TEST/"
SAMPLE_NUM = 100 # max sample num is 347
TOKENIZER = BartTokenizer.from_pretrained("facebook/bart-base", add_prefix_space=True)
BART_MODEL = BartModel.from_pretrained("facebook/bart-base", output_hidden_states=True)