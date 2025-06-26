"""
Module for analyzing token probabilities in a language model output.
This module provides functionality to calculate the probability of generating specific sequences.
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TokenInfo:
    """Information about a single token's generation probability."""
    token: int
    token_text: str
    original_logit: float
    modified_logit: float
    position: int


def setup_model(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    """
    Initialize the model, tokenizer and determine the device.
    
    Args:
        model_name: Name or path of the model to load
        
    Returns:
        Tuple of (model, tokenizer, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {model_name}")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16
    ).to(device)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
    return model, tokenizer, device


def calculate_sequence_probability(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    modification_factor: float = 0.5
) -> Tuple[float, List[TokenInfo]]:
    """
    Calculate the probability of generating a specific sequence of tokens.
    
    Args:
        text: The text to analyze
        model: The language model
        tokenizer: The tokenizer
        device: The device to run calculations on
        modification_factor: Factor to modify logits by (default: 0.5)
        
    Returns:
        Tuple containing:
        - total_log_prob: The log probability of the entire sequence
        - token_infos: List of TokenInfo objects containing token-wise information
    """
    # Tokenize the input text
    tokens = tokenizer.encode(text)
    token_infos: List[TokenInfo] = []
    total_log_prob = 0.0
    
    # Process each token in sequence
    for i, target_token in enumerate(tokens):
        # Get the input sequence up to the current position
        input_tokens = tokens[:i + 1]
        encoded = tokenizer(
            tokenizer.decode(input_tokens),
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        # Get model output
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
        # Get logits and calculate probabilities
        logits = outputs.logits[0, -1, :]
        original_logit = logits[target_token].item()
        modified_logit = original_logit * modification_factor
        
        # Store token information
        token_info = TokenInfo(
            token=target_token,
            token_text=tokenizer.decode([target_token]),
            original_logit=original_logit,
            modified_logit=modified_logit,
            position=i
        )
        token_infos.append(token_info)
        
        # Update total log probability using modified logits
        probs = F.softmax(logits * modification_factor, dim=-1)
        total_log_prob += torch.log(probs[target_token]).item()
    
    return total_log_prob, token_infos


def print_sequence_analysis(text: str, total_log_prob: float, token_infos: List[TokenInfo]) -> None:
    """
    Print a detailed analysis of the sequence probabilities.
    
    Args:
        text: The analyzed text
        total_log_prob: Total log probability of the sequence
        token_infos: List of TokenInfo objects with token-wise information
    """
    print(f"\nAnalyzing text: {text}")
    print(f"Total log probability: {total_log_prob:.4f}")
    print(f"Sequence probability: {torch.exp(torch.tensor(total_log_prob)):.4e}")
    
    # Print token-by-token analysis
    print("\nToken-by-token analysis:")
    print("-" * 100)
    headers = ["Position", "Token", "Text", "Original Logit", "Modified Logit"]
    print(f"{headers[0]:<8} {headers[1]:<8} {headers[2]:<15} {headers[3]:<15} {headers[4]:<15}")
    print("-" * 100)
    
    for info in token_infos:
        print(
            f"{info.position:<8} {info.token:<8} {info.token_text:<15} "
            f"{info.original_logit:>14.4f} {info.modified_logit:>14.4f}"
        )


def main() -> None:
    """Main function to demonstrate sequence probability calculation."""
    model_name = "speakleash/Bielik-7B-Instruct-v0.1"
    model, tokenizer, device = setup_model(model_name)
    
    example_text = (
        "Odpowiadaj krótko, precyzyjnie i wyłącznie w języku polskim. "
        "Pogoda w Warszawie jest dobra."
    )
    
    total_log_prob, token_infos = calculate_sequence_probability(
        example_text, model, tokenizer, device
    )
    print_sequence_analysis(example_text, total_log_prob, token_infos)


if __name__ == "__main__":
    main()
