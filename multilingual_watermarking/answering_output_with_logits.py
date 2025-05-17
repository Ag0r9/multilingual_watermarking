import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

model_name = "speakleash/Bielik-7B-Instruct-v0.1"

print(f"Loading model: {model_name}")
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

beginning_prompt = "Odpowiadaj krótko, precyzyjnie i wyłącznie w języku polskim. "

prompt_text = beginning_prompt + "Pogoda w Warszawie jest"

# Set pad token if not present (common for GPT-2)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Tokenize the prompt with padding and attention mask
encoded_prompt = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True)
input_ids = encoded_prompt["input_ids"].to(device)
attention_mask = encoded_prompt["attention_mask"].to(device)

def generate_output_with_logits(model, generated, attention_mask):
    """
    Generate output from the model and return logits.
    """
    outputs = model(input_ids=generated, attention_mask=attention_mask)
    next_token_logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
    generated = torch.cat([generated, next_token_id], dim=-1)
    # Update attention mask
    attention_mask = torch.cat(
        [attention_mask, torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)],
        dim=1
    )
    return generated, attention_mask, next_token_id, next_token_logits
    

# Generate tokens step by step in a loop
generated = input_ids
model.eval()
with torch.no_grad():
    for _ in range(100):
        generated, attention_mask, next_token_id, next_token_logits = generate_output_with_logits(model, generated, attention_mask)
        next_token_logits = next_token_logits / 2
        # Stop if EOS token is generated
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Next token ID: {next_token_id.item()}")
        print(f"Next token logits: {next_token_logits}")
        print(f"Generated text so far: {output_text}")

# Decode and print the generated text
output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
print(output_text)