import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F


def calculate_sequence_probability(text: str, model, tokenizer, device) -> tuple[float, list[dict]]:
    #TODO: zamiast probability zwrócić logit
    #TODO: logitsy pomiędzy check_logits a answer_with_logits są różne. Sprawdź dlaczego
    """
    Calculate the probability of generating a specific sequence of tokens and return token-wise probabilities.
    
    Args:
        text: The text to analyze
        model: The language model
        tokenizer: The tokenizer
        device: The device to run calculations on
        
    Returns:
        tuple containing:
        - total_log_prob: The log probability of the entire sequence
        - token_probs: List of dictionaries containing token-wise information
    """
    # Tokenize the input text
    tokens = tokenizer.encode(text)
    token_probs = []
    total_log_prob = 0.0
    
    # For each position in the sequence
    for i in range(len(tokens)):
        # Get the input sequence up to the current position
        input_tokens = tokens[:i+1]
        input_ids = torch.tensor([input_tokens]).to(device)
        
        # Get model output
        with torch.no_grad():
            outputs = model(input_ids)
            
        # Get logits for the last token
        logits = outputs.logits[0, -1, :]
        
        # Convert logits to probabilities using softmax
        probs = F.softmax(logits, dim=-1)
        
        # Get probability of the actual next token
        target_token = tokens[i]
        token_prob = probs[target_token].item()
        token_log_prob = torch.log(probs[target_token]).item()
        
        # Add to total log probability
        total_log_prob += token_log_prob
        
        # Store token information
        token_info = {
            'token': target_token,
            'token_text': tokenizer.decode([target_token]),
            'probability': token_prob,
            'log_probability': token_log_prob,
            'position': i
        }
        token_probs.append(token_info)
    
    return total_log_prob, token_probs

def print_sequence_analysis(text: str, total_log_prob: float, token_probs: list[dict]):
    """Print a detailed analysis of the sequence probabilities."""
    print(f"\nAnalyzing text: {text}")
    print(f"Total log probability: {total_log_prob:.4f}")
    print(f"Sequence probability: {torch.exp(torch.tensor(total_log_prob)):.4e}")
    print("\nToken-by-token analysis:")
    print("-" * 80)
    print(f"{'Position':<10} {'Token':<15} {'Text':<15} {'Probability':<15} {'Log Prob':<15}")
    print("-" * 80)
    
    for info in token_probs:
        print(f"{info['position']:<10} {info['token']:<15} {info['token_text']:<15} {info['probability']:.4e} {info['log_probability']:.4f}")

# Example usage

model_name = "speakleash/Bielik-7B-Instruct-v0.1"

print(f"Loading model: {model_name}")
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
example_text ="Odpowiadaj krótko, precyzyjnie i wyłącznie w języku polskim. Pogoda w Warszawie jest taka sama jak w całym kraju. W Polsce w lipcu i sierpniu jest ciepło, a nawet gorąco. W lipcu temperatury wahają się od 16°C do 31°C. W sierpniu temperatury wahają się od 17°C do 30°C."
total_log_prob, token_probs = calculate_sequence_probability(example_text, model, tokenizer, device)
print_sequence_analysis(example_text, total_log_prob, token_probs)





