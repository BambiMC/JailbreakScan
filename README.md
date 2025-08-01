# JailbreakScan

PromptGuard is a modular, production-ready benchmark designed to detect jailbreak vulnerabilities in existing prompts that query language models. PromptGuard is fully compatible with any Huggingface model, making it a flexible component for evaluating prompt safety across a wide range of architectures and deployments.

## Installation

```bash
pip install transformers datasets openai accelerate bitsandbytes
```

## Usage

```bash
python JailbreakScan.py --model_name meta-llama/Llama-2-7b-chat-hf --max_examples 10
```
