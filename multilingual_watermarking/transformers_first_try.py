import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "speakleash/Bielik-7B-Instruct-v0.1"

print(f"Loading model: {model_name}")
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

messages = [
    {"role": "system", "content": "Odpowiadaj krótko, precyzyjnie i wyłącznie w języku polskim."},
    {"role": "user", "content": "Jakie mamy pory roku w Polsce?"},
    {"role": "assistant", "content": "W Polsce mamy 4 pory roku: wiosna, lato, jesień i zima."},
    {"role": "user", "content": "Która jest najcieplejsza?"},
]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = input_ids.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])
