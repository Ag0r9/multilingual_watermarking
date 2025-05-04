import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from multilingual_watermarking.ForceStringLogitsProcessor import ForceStringLogitsProcessor

model_name = "speakleash/Bielik-7B-Instruct-v0.1"

print(f"Loading model: {model_name}")
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)


# --- Configure Generation ---
prompt_text = "Pogoda w Warszawie jest"
# The string we want to force *after* the prompt
target_string_to_force = " słoneczna i cudowna. Wspaniały dzień na spacer."

# Set pad token if not present (common for GPT-2)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Tokenize the prompt with padding and attention mask
encoded_prompt = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True)
input_ids = encoded_prompt["input_ids"].to(device)
attention_mask = encoded_prompt["attention_mask"].to(device)
prompt_length = input_ids.shape[1]

# Tokenize the prompt and forced string separately for debugging
prompt_tokens = tokenizer.tokenize(prompt_text)
forced_tokens = tokenizer.tokenize(target_string_to_force)

print("\n--- Debugging Tokenization ---")
print(f"Prompt Tokens: {prompt_tokens}")
print(f"Forced Tokens: {forced_tokens}")

# Calculate the correct start index based on the prompt length
start_index = len(prompt_tokens) - 1

# --- Create the Logits Processor ---
# We want to start forcing immediately after the prompt
force_processor = ForceStringLogitsProcessor(
    target_string=target_string_to_force,
    tokenizer=tokenizer,
    start_index=start_index,  # Start forcing after the prompt tokens
)

# Wrap it in a list (you can add other processors here too)
logits_processor_list = LogitsProcessorList([force_processor])

# --- Generate Text ---
# Ensure max_new_tokens is long enough to accommodate the forced string
# and potentially some free generation afterwards.
target_token_count = len(force_processor.target_token_ids)
min_required_length = target_token_count
max_length_to_generate = prompt_length + min_required_length + 20  # Force + 20 extra tokens

print("\nStarting generation...")
print(f"Prompt: '{prompt_text}'")
print(f"Forcing: '{target_string_to_force}' (length {target_token_count} tokens)")
print(f"Minimum required new tokens: {min_required_length}")
print(f"Max total length: {max_length_to_generate}")

# Generate using the custom processor
output_sequences = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,  # Pass the attention mask
    logits_processor=logits_processor_list,
    max_length=max_length_to_generate,  # Use max_length which includes prompt
    eos_token_id=tokenizer.eos_token_id,  # Stop generation if EOS is forced or generated later
    pad_token_id=tokenizer.pad_token_id,  # Explicitly set the pad token ID
)

# --- Decode and Print ---
generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

print("\n--- Generated Output ---")
print(generated_text)

# Verify the forced part
expected_start = prompt_text + target_string_to_force
if generated_text.startswith(expected_start):
    print("\nVerification: Forced string successfully generated at the beginning.")
else:
    print("\nVerification Failed: Output does not start with the expected forced string.")
    print(f"Expected start: '{expected_start}'")
