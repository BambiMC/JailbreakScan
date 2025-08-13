



# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText

import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    Llama4ForConditionalGeneration,
    Mistral3ForConditionalGeneration,
    AutoProcessor,
)
from datasets import load_dataset
from tqdm import tqdm
import torch
import torch.nn.functional as F
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
import judge_model 



# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
# model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")
# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# inputs = tokenizer.apply_chat_template(
# 	messages,
# 	add_generation_prompt=True,
# 	tokenize=True,
# 	return_dict=True,
# 	return_tensors="pt",
# ).to(model.device)

# outputs = model.generate(**inputs, max_new_tokens=8)
# print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))