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



class BaseModel:
    def __init__(self, model_name, load_in_4bit=False, multi_gpu=False):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.multi_gpu = multi_gpu

        self.model = None
        self.tokenizer = None


    def load_model(self, model_name, load_in_4bit=False, multi_gpu=False):
        raise NotImplementedError("Bitte in der Unterklasse implementieren.")
    def generate_batch_responses(self, prompts, tokenizer, model, max_new_tokens):
        raise NotImplementedError("Bitte in der Unterklasse implementieren.")


# class Llama4(BaseModel):
#     def __init__(self, model_name, load_in_4bit=False, multi_gpu=False):
#         super().__init__(model_name, load_in_4bit=False, multi_gpu=False)


#     def load_model(self, model_name, load_in_4bit=False, multi_gpu=False):
#         print(f"Llama4 ({self.model_name}) wird geladen...")

#         bnb_config = None
#         if load_in_4bit:
#             bnb_config = BitsAndBytesConfig(
#                 load_in_4bit=load_in_4bit,
#                 bnb_4bit_use_double_quant=True,
#                 bnb_4bit_quant_type="nf4",
#                 bnb_4bit_compute_dtype=torch.bfloat16,
#         )

#         processor = AutoProcessor.from_pretrained(model_name)
#         model = AutoModelForImageTextToText.from_pretrained(
#             model_name,
#             torch_dtype=torch.float16,
#             device_map="auto" if multi_gpu else None,
#             quantization_config=bnb_config
#         )
#         # messages = [
#         #     {
#         #         "role": "user",
#         #         "content": [
#         #             {"type": "text", "text": "What animal is on the candy?"}
#         #         ]
#         #     },
#         # ]
#         # inputs = processor.apply_chat_template(
#         #     messages,
#         #     add_generation_prompt=True,
#         #     tokenize=True,
#         #     return_dict=True,
#         #     return_tensors="pt",
#         # ).to(model.device)
#         # outputs = model.generate(**inputs, max_new_tokens)
#         # decoded = processor.decode(
#         #     outputs[0][inputs["input_ids"].shape[-1]:],
#         #     skip_special_tokens=True
#         # )
#         # print(decoded)

#         return model

#     def generate_batch_responses(self, prompts, max_new_tokens):
#         print(f"Llama4 ({self.model_name}) generiert Antworten...")
#         inputs = self.tokenizer(
#             prompts,
#             return_tensors="pt",
#             padding=True,
#             truncation=True
#         ).to(self.model.device)

#         outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
#         decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         return decoded

class Llama4(BaseModel):
    def __init__(self, model_name, load_in_4bit=False, multi_gpu=False):
        super().__init__(model_name, load_in_4bit=False, multi_gpu=False)

    def load_model(self, model_name, load_in_4bit=False, multi_gpu=False):
        print(f"Llama4 ({self.model_name}) wird geladen...")

        # Default to the official LLaMA-4 Scout model if not provided
        if model_name is None:
            model_name = "meta-llama/Llama-4-Scout-17B-16E"

        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

        # Optional: Adjust processor settings if needed
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        # Model loading
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    def generate_batch_responses(self, batch_messages, max_new_tokens=40):
        """
        batch_messages: list of chat message lists (each element is a messages array like in the Hugging Face docs)
        Example:
        [
            [
                {"role": "user", "content": [
                    {"type": "image", "url": "https://example.com/image.jpg"},
                    {"type": "text", "text": "What's in the picture?"}
                ]}
            ],
            ...
        ]
        """
        print(f"Llama4Scout ({self.model_name}) generiert Antworten...")

        results = []
        for messages in batch_messages:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            decoded = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
            results.append(decoded.strip())

        return results

class Gemma3n(BaseModel):
    def __init__(self, model_name, load_in_4bit=False, multi_gpu=False):
        super().__init__(model_name, load_in_4bit=False, multi_gpu=False)


    def load_model(self, model_name, load_in_4bit=False, multi_gpu=False):
        print(f"Gemma3n ({self.model_name}) wird geladen...")

        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E4B-it", use_fast=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("⚠️ Detected gemma-3n — forcing single GPU (cuda:0) to avoid cross-device tensor mismatch.")
        self.model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3n-E4B-it",
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            # quantization_config=bnb_config,
        )

    def generate_batch_responses(self, prompts, max_new_tokens):
        print(f"Gemma3n ({self.model_name}) generiert Antworten...")
        return generate_batch_responses_generic(self, prompts, max_new_tokens)


class Mistral3(BaseModel):
    def __init__(self, model_name, load_in_4bit=False, multi_gpu=False):
        super().__init__(model_name, load_in_4bit=False, multi_gpu=False)


    def load_model(self, model_name, load_in_4bit=False, multi_gpu=False):
        print(f"Mistral3 ({self.model_name}) wird geladen...")
        tokenizer = MistralTokenizer.from_hf_hub(model_name)
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto" if multi_gpu else None,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )
        return tokenizer, model

    def generate_batch_responses(self, prompts, max_new_tokens):
        responses = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            request = ChatCompletionRequest(messages=messages)
            tokenized = self.tokenizer.encode_chat_completion(request)
            input_ids = torch.tensor([tokenized.tokens]).to(self.model.device)
            attention_mask = torch.ones_like(input_ids)

            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
            )[0]

            decoded = self.tokenizer.decode(output[len(tokenized.tokens):])
            responses.append(decoded)
        return responses

class OpenAI_OSS(BaseModel):
    def __init__(self, model_name, load_in_4bit=False, multi_gpu=False):
        super().__init__(model_name, load_in_4bit=False, multi_gpu=False)


    def load_model(self, model_name, load_in_4bit=False, multi_gpu=False):
        print(f"OpenAI_OSS ({self.model_name}) wird geladen...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("multi_gpu: ", multi_gpu)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if multi_gpu else None,
            torch_dtype=torch.bfloat16,
        )

    def generate_batch_responses(self, prompts, max_new_tokens):
        return generate_batch_responses_generic(self, prompts, max_new_tokens)

class FallbackModel(BaseModel):
    def __init__(self, model_name, load_in_4bit=False, multi_gpu=False):
        super().__init__(model_name, load_in_4bit=False, multi_gpu=False)


    def load_model(self, model_name, load_in_4bit=False, multi_gpu=False):
        print(f"FallbackModel: unbekanntes Modell '{self.model_name}', Standardverhalten wird genutzt.")

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" if multi_gpu else None,
            torch_dtype=torch.bfloat16,
        )

    def generate_batch_responses(self, prompts, max_new_tokens):
        return generate_batch_responses_generic(self, prompts, max_new_tokens)



def generate_batch_responses_generic(self, prompts, max_new_tokens):
    if self.model is None or self.tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    print(f"FallbackModel ({self.model_name}) generiert Antworten...")

    # Fallback to legacy generation
    formatted_prompts = [format_prompt(p, self.tokenizer) for p in prompts]
    inputs = self.tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

    outputs = self.model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        pad_token_id=self.tokenizer.pad_token_id,
        eos_token_id=self.tokenizer.eos_token_id,
    )
    print(f"Generated output: {self.tokenizer.batch_decode(outputs, skip_special_tokens=True)}")
    return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)



class HF_Model:
    _patterns = [
        ("llama4", Llama4),
        ("gemma-3n", Gemma3n),
        ("mistral-large", Mistral3),
        ("openai/gpt-oss", OpenAI_OSS),
        # ("", ),
        # ("", ),
        # ("", ),
        # ("", ),
    ]
    _fallback_class = FallbackModel

    def __new__(cls, model_name, *args, **kwargs):
        # Match model_name against patterns
        for pattern, model_class in cls._patterns:
            if pattern in model_name.lower():
                return model_class(model_name, *args, **kwargs)
        # use fallback class
        return cls._fallback_class(model_name, *args, **kwargs)



def strip_input_from_output(outputs, inputs):
    stripped = []
    for output, prompt in zip(outputs, inputs):
        if output.startswith(prompt):
            stripped.append(output[len(prompt):])
        else:
            idx = output.find(prompt)
            if idx != -1:
                stripped.append(output[idx + len(prompt):])
            else:
                stripped.append(output)
    return stripped


# === Main Script ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--use-judge-model", action="store_true", help="Use a judge model for evaluation")
    parser.add_argument("--judge_model", type=str, required=False, help="HF model ID for the judge model")
    parser.add_argument("--max_examples", type=int, default=None, help="Number of examples to evaluate")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generation")

    args = parser.parse_args()

    if args.use_judge_model and not args.judge_model:
        raise ValueError("--judge_model must be provided if --use-judge-model is used.")
    if args.judge_model and not args.use_judge_model:
        raise ValueError("Use --use-judge-model to enable judge model evaluation.")
    if args.use_judge_model:
        judge_model.load_judge_model(args.judge_model)

    model = HF_Model(args.model_name)
    model.load_model(args.model_name, load_in_4bit=True, multi_gpu=args.multi_gpu)

    # thinking_models = ["openai/gpt-oss-20b", "zai-org/GLM-4-32B-0414", "openai/gpt-oss-120b", "Qwen/Qwen3-32B", "Qwen/Qwen3-235B-A22B", "deepseek-ai/DeepSeek-R1-0528", "deepseek-ai/DeepSeek-V3-0324"]
    # if any(args.model_name.lower() in m.lower() for m in thinking_models):
    #     max_new_tokens = 1024
    # else:
    max_new_tokens = 4

    print("Loading AdvBench dataset...")
    dataset = load_dataset("walledai/advbench", split="train")
    end = args.end if args.end is not None else len(dataset)
    dataset = dataset.select(range(args.start, end))
    if args.max_examples:
        dataset = dataset.select(range(min(len(dataset), args.max_examples)))

    prompts = dataset["prompt"]
    results = []

    with tqdm(total=len(prompts), desc="Evaluating") as pbar:
        for batch_prompts in batched(prompts, args.batch_size):
            model_outputs = model.generate_batch_responses(batch_prompts, max_new_tokens)
            answers = strip_input_from_output(model_outputs, batch_prompts)
            if args.use_judge_model:
                verdicts = judge_model.evaluate_batch_with_judge(model_outputs, batch_prompts)
            else:
                verdicts = ["safe" if keyword_judge(output) else "jailbroken" for output in answers]
                print(f"Verdicts for batch: {verdicts}")


            for p, o, v in zip(batch_prompts, answers, verdicts):
                results.append({
                    "adv_prompt": p,
                    "model_output": o,
                    "verdict": v
                })

            pbar.update(len(batch_prompts))

    # Write results
    with open("jailbreak_scan_results.txt", "w") as f:
        f.write(f"-- Jailbreak Scan Results for {args.model_name} --\n")
        for r in results:
            f.write(f"Adv Prompt: {r['adv_prompt']}\n")
            f.write(f"Model Output: {r['model_output']}\n")
            f.write(f"Verdict: {r['verdict']}\n")
            f.write("----------------------------------------------\n")

    # Summary
    from collections import Counter
    summary = Counter([r["verdict"] for r in results])

    print("\n=== Evaluation Summary ===")
    total = len(results)
    for label, count in summary.items():
        print(f"{label.capitalize()}: {count} ({count / total:.2%})")

if __name__ == "__main__":
    main()
