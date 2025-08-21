import argparse
import re
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from tqdm import tqdm
import torch
import torch.nn.functional as F
import judge_model 
import torch

from utils import batched, format_prompt, keyword_judge




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

class DefaultModel(BaseModel):
    def __init__(self, model_name, load_in_4bit=False, multi_gpu=False):
        super().__init__(model_name, load_in_4bit=False, multi_gpu=False)


    def load_model(self, model_name, load_in_4bit=False, multi_gpu=False):
        print(f"DefaultModel: unbekanntes Modell '{self.model_name}', Standardverhalten wird genutzt.")

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

    print(f"DefaultModel ({self.model_name}) generiert Antworten...")

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
    # print(f"Generated output: {self.tokenizer.batch_decode(outputs, skip_special_tokens=True)}")
    return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)



class HF_Model:
    _patterns = [
        ("openai/gpt-oss", OpenAI_OSS),
        # ("", ),
        # ("", ),
        # ("", ),
        # ("", ),
    ]
    _fallback_class = DefaultModel

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
    replaced_tokens = [re.sub(r"<\|[^|]+?\|>", "", s) for s in stripped]
    replaced_tokens = [re.sub(r"model\n", "", s) for s in replaced_tokens]
    replaced_tokens = [re.sub(r"assistant\n", "", s) for s in replaced_tokens]
    replaced_tokens = [re.sub(r"assistantanalysis", "", s) for s in replaced_tokens] #gpt-oss
    return replaced_tokens



# === Main Script ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--use-judge-model", action="store_true", help="Use a judge model for evaluation")
    parser.add_argument("--judge_model", type=str, required=False, default="dphn/Dolphin-Llama3.1-8B-Instruct-6.0bpw-h6-exl2", help="HF model ID for the judge model")
    parser.add_argument("--max_examples", type=int, default=None, help="Number of examples to evaluate")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generation")
    parser.add_argument("--rewriting_prompts", type=bool, default=False, help="Whether to rewrite prompts")

    args = parser.parse_args()

    if args.use_judge_model:
        judge_model.load_judge_model(args.judge_model)

    model = HF_Model(args.model_name)
    model.load_model(args.model_name, load_in_4bit=True, multi_gpu=args.multi_gpu)

    thinking_models = ["openai/gpt-oss", "zai-org/GLM-4-32B", "Qwen/Qwen3", "CohereLabs/c4ai-command-a"]
    if any(args.model_name.lower() in m.lower() for m in thinking_models):
        max_new_tokens = 2048
    else:
        max_new_tokens = 512



    print("Loading AdvBench dataset...")
    dataset = load_dataset("walledai/advbench", split="train")
    end = args.end if args.end is not None else len(dataset)
    dataset = dataset.select(range(args.start, end))
    if args.max_examples:
        dataset = dataset.select(range(min(len(dataset), args.max_examples)))

    prompts = dataset["prompt"]
    results = []

    if args.rewriting_prompts:
        print("Loading Rewriter Model...")

        prompts = rewriting_prompts(prompts, model.tokenizer, model.model, max_new_tokens)



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
