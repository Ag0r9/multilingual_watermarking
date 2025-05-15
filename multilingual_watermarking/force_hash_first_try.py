from transformers import AutoTokenizer
from HashForceWatermarkDetector import HashForceWatermarkDetector

# Initialize
model_name = "speakleash/Bielik-7B-Instruct-v0.1"

print(f"Loading model: {model_name}")
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
detector = HashForceWatermarkDetector(tokenizer)

# Example usage with generated text
generated_text = "Your generated text with watermark"
secret_key = "optional_secret_key"  # Same key used during generation

# Analyze the text
results = detector.analyze_text(generated_text, secret_key)

if results["watermark_detected"]:
    print("Watermark found!")
    print(f"Integrity verified: {results['integrity_verified']}")
    print("Found watermarks:", results["found_hashes"])
    
    if results["integrity_verified"]:
        print("Text has not been modified since watermarking")
    else:
        print("Warning: Text may have been modified!")
else:
    print("No watermark detected")