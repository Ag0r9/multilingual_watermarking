from multilingual_watermarking.base_watermarking.watermark_processor import WatermarkDetector
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

with open("output_text.txt", "r", encoding="utf-8") as file:
    output_text = file.read()

model_name = "speakleash/Bielik-7B-Instruct-v0.1"
expand_output = True

print(f"Loading model: {model_name}")
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=0.25, # should match original setting
                                        seeding_scheme="selfhash", # should match original setting
                                        device=model.device, # must match the original rng device type
                                        tokenizer=tokenizer,
                                        z_threshold=4.0,
                                        normalizers=[],
                                        ignore_repeated_ngrams=True)

score_dict = watermark_detector.detect(output_text)

print(score_dict)