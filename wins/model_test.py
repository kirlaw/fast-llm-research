#!/usr/bin/python3
# author Eloise

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "E:/Qwen3-1.7B"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# 加载模型（低显存优化）
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",         # 自动选择 GPU
    dtype=torch.float16, # 半精度
    #load_in_8bit=True          # 使用 bitsandbytes 8bit 量化
)

# 测试推理
prompt = "你好，请用中文介绍一下你自己。"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
