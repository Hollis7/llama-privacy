from transformers import LlamaForCausalLM, LlamaTokenizer
model_path = "llama-2-7b-hf"
base_tokenizer = LlamaTokenizer.from_pretrained(model_path)
base_model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto", load_in_8bit=True)
