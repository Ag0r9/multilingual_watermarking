import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from multilingual_watermarking.logit_modification import (
    LogitModificationTracker,
    generate_output_with_logits,
)
from multilingual_watermarking.paths import Paths


def calculate_sequence_probability(
    text: str, model, tokenizer, device
) -> tuple[float, list[dict]]:
    """
    Calculate the probability of generating a specific sequence of tokens and return token-wise probabilities.

    Args:
        text: The text to analyze
        model: The language model
        tokenizer: The tokenizer
        device: The device to run calculations on
    """
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Tokenize the input text with padding and attention mask
    tokens = tokenizer.encode(text)
    tracker = LogitModificationTracker()

    # For each position in the sequence
    for i in range(len(tokens)):
        # Get the input sequence up to the current position
        input_tokens = tokens[: i]
        encoded = tokenizer(
            tokenizer.decode(input_tokens), return_tensors="pt", padding=True, truncation=True
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            
        # Get logits for the last token
        logits = outputs.logits[0, -1, :]
        
        # Get logit of the actual next token
        target_token = tokens[i]
        
        tracker.add_token_info(
            token_id=target_token,
            original_logits=logits[target_token],
            modified_logits=logits[target_token],
            position=i,
        )
    
    tracker.save_history_to_csv()
    return


# Example usage
if __name__ == "__main__":
    model_name = "speakleash/Bielik-7B-Instruct-v0.1"

    paths = Paths()

    print(f"Loading model: {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

    with open(paths.GENERATED_TEXT_DIR / "output.txt", "r", encoding="utf-8") as f:
        example_text = f.read()
    calculate_sequence_probability(
        example_text, model, tokenizer, device
    )
