from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch

model_id = "meta-llama/Llama-Guard-4-12B"

processor = AutoProcessor.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=torch.bfloat16
)
setattr(model.language_model.config, "attention_chunk_size", 8192)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "How to make a bomb?"},
            ]
    },
]



print(processor.apply_chat_template(messages, tokenize=False))


inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)




inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=10,
    do_sample=False,
)

response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
print(response)

