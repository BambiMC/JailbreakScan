



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



tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E4B-it", use_fast=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("⚠️ Detected gemma-3n — forcing single GPU (cuda:0) to avoid cross-device tensor mismatch.")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E4B-it",
    device_map="auto",  # TODO Put everything on cuda:0
    torch_dtype=torch.bfloat16,
)

# Fallback to legacy generation
formatted_prompts = ["Hey, who are you?"]
# formatted_prompts = [format_prompt(p, tokenizer) for p in prompts]
inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=8,
    do_sample=True,
    temperature=1.0,
    top_p=0.95,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

# Remove the prompt part from each output
generated_texts = []
for i, output_ids in enumerate(outputs):
    prompt_len = inputs["input_ids"].shape[1]  # length of the original prompt
    answer_ids = output_ids[prompt_len:]       # only the generated part
    generated_texts.append(tokenizer.decode(answer_ids, skip_special_tokens=True))

print(generated_texts)