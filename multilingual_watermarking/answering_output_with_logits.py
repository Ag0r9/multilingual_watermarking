import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

from multilingual_watermarking.logit_modification import (
    LogitModificationTracker,
    generate_output_with_logits,
)
from multilingual_watermarking.paths import Paths

promts = [
    "Napisz jeden paragraf tekstu na temat historii pizzy.",
    "Napisz jeden paragraf tekstu na temat Wielkiego Muru Chińskiego.",
    "Napisz jeden paragraf tekstu na temat wpływu muzyki na nastrój.",
    "Napisz jeden paragraf tekstu na temat życia pszczół w ulu.",
    "Napisz jeden paragraf tekstu na temat Układu Słonecznego.",
    "Napisz jeden paragraf tekstu na temat korzyści płynących z regularnego czytania książek.",
    "Napisz jeden paragraf tekstu na temat funkcjonowania silnika spalinowego.",
    "Napisz jeden paragraf tekstu na temat procesu fotosyntezy.",
    "Napisz jeden paragraf tekstu na temat starożytnego Egiptu.",
    "Napisz jeden paragraf tekstu na temat znaczenia wody dla życia na Ziemi.",
    "Napisz jeden paragraf tekstu na temat tego, jak powstają wulkany.",
    "Napisz jeden paragraf tekstu na temat historii igrzysk olimpijskich.",
    "Napisz jeden paragraf tekstu na temat budowy i funkcji ludzkiego serca.",
    "Napisz jeden paragraf tekstu na temat Internetu i jego wpływu na społeczeństwo.",
    "Napisz jeden paragraf tekstu na temat przyczyn i skutków zmiany klimatu.",
    "Napisz jeden paragraf tekstu na temat życia i twórczości Marii Skłodowskiej-Curie.",
    "Napisz jeden paragraf tekstu na temat Amazonii, największego lasu deszczowego na świecie.",
    "Napisz jeden paragraf tekstu na temat rozwoju sztucznej inteligencji.",
    "Napisz jeden paragraf tekstu na temat cyklu życia motyla.",
    "Napisz jeden paragraf tekstu na temat roli witamin w diecie człowieka.",
]

model_name = "speakleash/Bielik-7B-Instruct-v0.1"

print(f"Loading model: {model_name}")
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

paths = Paths()

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

beginning_prompt = "Odpowiadaj krótko, precyzyjnie i wyłącznie w języku polskim. "

prompt_text = promts[0]

prompt_text = beginning_prompt + prompt_text

# Set pad token if not present (common for GPT-2)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Tokenize the prompt with padding and attention mask
encoded_prompt = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True)
input_ids = encoded_prompt["input_ids"].to(device)
attention_mask = encoded_prompt["attention_mask"].to(device)

# Initialize the tracker
tracker = LogitModificationTracker()

# Generate tokens step by step in a loop
generated = input_ids
model.eval()
with torch.no_grad():
    for pos in range(1000):
        generated, attention_mask, next_token_id, modified_logits = generate_output_with_logits(
            model,
            device,
            generated,
            attention_mask,
            tracker,
            position=pos,
        )

        # Stop if EOS token is generated
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"\nGenerated text so far: {output_text}")

        # Print token information
        token_info = tracker.token_history[-1]
        print(f"Token ID: {token_info['token_id']}")
        print(f"Original logit: {token_info['original_logits'].max().item():.4f}")
        print(f"Modified logit: {token_info['modified_logits'].max().item():.4f}")

# Final output
output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
print("\nFinal output:", output_text)
file_name = paths.GENERATED_TEXT_DIR / "output.txt"
# Save the output to a file
# Ensure the directory exists
paths.GENERATED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
with open(file_name, "w", encoding="utf-8") as f:
    f.write(output_text)
print("Output saved to ", file_name)
# Save the logits to a CSV file
tracker.save_history_to_csv()
