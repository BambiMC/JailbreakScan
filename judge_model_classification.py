from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch

judge_processor = None
judge_model = None

def load_judge_model(model_id: str = "meta-llama/Llama-Guard-4-12B"):
    global judge_processor, judge_model
    print(f"Loading judge model: {model_id}")


    model_id = "meta-llama/Llama-Guard-4-12B"

    judge_processor = AutoProcessor.from_pretrained(model_id)
    judge_model = Llama4ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )

    setattr(judge_model.language_model.config, "attention_chunk_size", 8192)


    print("Patched attention_chunk_size = 512")


# === Evaluate Responses with Judge ===
def evaluate_batch_with_judge(model_outputs, prompts):
    labels = []

    for prompt, response in zip(prompts, model_outputs):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": response}]},
        ]

        inputs = judge_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        # Ensure all tensors are on the correct device
        inputs = {k: v.to(judge_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = judge_model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )

        classification = judge_processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )[0].strip().lower()

        labels.append(classification)

    return labels