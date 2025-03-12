# %%
from datasets import load_dataset
import torch,einops
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# %%
import csv
from transformers import TrainerCallback

class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_file="loss_log2.csv"):
        self.log_file = log_file
        # 创建 CSV 文件并写入表头
        with open(self.log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        """记录损失到 CSV 文件"""
        if logs is not None and "loss" in logs:
            step = state.global_step
            loss = logs["loss"]

            # 追加到 CSV
            with open(self.log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, loss])

            print(f"Step {step}: Loss = {loss}")  # 终端打印损失

# %%
dataset = load_dataset("json",data_files="/home/user/hdb/projects/pyProject/llama-privacy/data-tmp/Belle_open_source_0.5M_changed_test2.json",split="train")
dataset = dataset.filter(lambda x: x["text"] is not None and len(x["text"].strip()) > 0)
# %%

base_model_name ="llama-2-7b-chat-hf"
device_map = {"": 0}  # 只使用 GPU 0
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # 改为 bfloat16
)


# %%
# 加载本地模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,#本地模型名称
    quantization_config=bnb_config,#上面本地模型的配置
    device_map=device_map,#使用GPU的编号
    trust_remote_code=True,
    use_auth_token=True
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1
# 确保所有模型参数都在 `cuda:0`
# 确保 pretraining_tp 兼容
if hasattr(base_model.config, "pretraining_tp"):
    base_model.config.pretraining_tp = 1

print(base_model.device)  # 确保模型正确加载到 GPU
# %%
#参数是啥意思我也不知道，大家会调库就行了
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
# %%
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# %%
output_dir = "./results2"
training_args = TrainingArguments(
    report_to="none",  # 禁用 wandb 日志记录
    output_dir=output_dir,#训练后输出目录
    per_device_train_batch_size=1,#每个GPU的批处理数据量
    gradient_accumulation_steps=4,#在执行反向传播/更新过程之前，要累积其梯度的更新步骤数
    learning_rate=2e-4,#超参、初始学习率。太大模型不稳定，太小则模型不能收敛
    logging_steps=10,#两个日志记录之间的更新步骤数
    num_train_epochs=3,  # 训练 3 个完整 epoch

)
max_seq_length = 1024
#TrainingArguments 的参数详解：https://blog.csdn.net/qq_33293040/article/details/117376382

trainer = SFTTrainer(
    model=base_model.to("cuda:0"),  # 强制模型在 GPU 0
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
    callbacks=[LossLoggerCallback("loss_log2.csv")],  # 添加损失记录回调
)
# %%
trainer.train()
# %%
output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
# %%

