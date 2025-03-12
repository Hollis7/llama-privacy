from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

#设置原来本地模型的地址
model_name_or_path = 'llama-2-7b-chat-hf'
#设置微调后模型的地址，就是上面的那个地址
adapter_name_or_path = 'results2/final_checkpoint'
#设置合并后模型的导出地址
save_path = 'data-tmp/new_model2'

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map='auto'
)
print("load model success")
model = PeftModel.from_pretrained(model, adapter_name_or_path)
print("load adapter success")
model = model.merge_and_unload()
print("merge success")

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print("save done.")
