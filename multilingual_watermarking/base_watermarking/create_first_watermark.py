from multilingual_watermarking.base_watermarking.watermark_processor import WatermarkLogitsProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import torch
import os

input = "Opowiedz mi o Powstaniu Wielkopolskim, które wydarzyło się w 1918 roku."
input_text = "Odpowiadaj krótko, precyzyjnie i wyłącznie w języku polskim." + input

model_name = "speakleash/Bielik-7B-Instruct-v0.1"
expand_output = True

print(f"Loading model: {model_name}")
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                               gamma=0.25,
                                               delta=2.0,
                                               seeding_scheme="selfhash") #equivalent to `ff-anchored_minhash_prf-4-True-15485863`
# Note:
# You can turn off self-hashing by setting the seeding scheme to `minhash`.

tokenized_input = tokenizer(input_text, return_tensors='pt').to(model.device)
# note that if the model is on cuda, then the input is on cuda
# and thus the watermarking rng is cuda-based.
# This is a different generator than the cpu-based rng in pytorch!

# Add generation parameters for longer output
generation_params = {
    "max_new_tokens": 512,  # Controls the maximum length of the generated text
    "min_new_tokens": 100,  # Ensures a minimum length of generated text
    "do_sample": True,      # Enables sampling-based generation
    "temperature": 0.5,     # Controls randomness (lower = more focused)
    "top_p": 0.9,
    "logits_processor": ([watermark_processor])
}

output_tokens = model.generate(**tokenized_input, **generation_params)

# if decoder only model, then we need to isolate the
# newly generated tokens as only those are watermarked, the input/prompt is not
output_tokens = output_tokens[:,tokenized_input["input_ids"].shape[-1]:]

output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

print("Output text:")
print(output_text)
print("Watermarking complete.")

with open("output_text.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

print("Output text saved")