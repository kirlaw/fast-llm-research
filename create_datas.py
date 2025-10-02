#!/usr/bin/python3
# author Eloise

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

teacher_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    device_map="auto",
    torch_dtype=torch.float16
)
teacher_model.eval()


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

prompts = [
    "介绍一下人工智能的发展历程。",
    "写一首关于春天的诗。",
    "解释一下量子计算的基本原理。",
]

def encode(example):
    inputs = tokenizer(example, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    return inputs

train_dataset = [encode(p) for p in prompts]