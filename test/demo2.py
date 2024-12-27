# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights
import torch
# %%
# 设置本地模型路径
local_model_path = "./llama-2-7b-hf"  # 你的本地模型文件夹路径

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
# 设置 pad_token_id，如果未设置则手动添加
tokenizer.pad_token = tokenizer.eos_token  # 使用 eos_token 作为 pad_token
# %%
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        device_map="auto",# 自动分配设备
        load_in_8bit=True
    )
# %%
# 定义输入文本
input_text = "Please make my age and address vague: Xiao Hong is 18 years old and I live in Chengdu to protect my privacy."
# 将输入文本编码为模型输入，并显式传递 attention_mask
inputs = tokenizer(
    input_text,
    return_tensors="pt",
    padding=True,  # 添加 padding
    truncation=True,
    max_length=512  # 设置输入最大长度
)
# 将输入和 attention_mask 移动到 GPU
inputs = {key: value.to("cuda") for key, value in inputs.items()}

# %%
# 生成文本（定义生成的最大长度）
max_length = 500
# 生成文本
output = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=500,
    num_return_sequences=1,
    temperature=0.7,
    top_k=50,
    top_p=0.9
)


# 解码生成的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
