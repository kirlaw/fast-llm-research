import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from create_datas import train_dataset


# 加载模型
teacher_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    device_map="auto",
    torch_dtype=torch.float16
)
teacher_model.eval()

student_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")


# 蒸馏损失函数
def distill_loss(student_logits, teacher_logits, temperature=2.0, alpha=0.5):
    """
    student_logits: 学生输出
    teacher_logits: 教师输出
    temperature: 蒸馏温度
    alpha: loss 权重
    """
    # 交叉熵 loss（学生拟合 ground truth）
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        teacher_logits.argmax(dim=-1).view(-1),
        ignore_index=-100
    )

    # 蒸馏 KL loss（学生拟合 teacher 分布）
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

    return alpha * kl_loss + (1 - alpha) * ce_loss

# --------------------------
# 4. 定义训练器
# --------------------------
class DistillTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        with torch.no_grad():
            teacher_out = teacher_model(**inputs).logits

        student_out = model(**inputs).logits
        loss = distill_loss(student_out, teacher_out)
        return (loss, student_out) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="./distill-qwen3",
    per_device_train_batch_size=1,          # batch小一些，显存不够就accumulation
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=True,                              # 半精度省显存
    optim="adamw_torch",
)

trainer = DistillTrainer(
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# --------------------------
# 5. 开始训练
# --------------------------
trainer.train()

# 保存蒸馏后的 student
trainer.save_model("./distill-qwen3-final")
