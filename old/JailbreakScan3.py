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
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

from utils import batched, format_prompt, keyword_judge

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import pipeline
import torch









# === Main Script ===
def main():
    # Load model directly
    from transformers import AutoProcessor, Gemma3nForConditionalGeneration
    from PIL import Image
    import requests
    import torch

    model_id = "google/gemma-3n-e4b-it"

    model = Gemma3nForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16,).eval()

    processor = AutoProcessor.from_pretrained(model_id)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
                {"type": "text", "text": "Describe this image in detail."}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)

    # **Overall Impression:** The image is a close-up shot of a vibrant garden scene,
    # focusing on a cluster of pink cosmos flowers and a busy bumblebee.
    # It has a slightly soft, natural feel, likely captured in daylight.


if __name__ == "__main__":
    main()