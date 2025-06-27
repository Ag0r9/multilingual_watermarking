"""
Module for generating text with logit tracking and watermarking capabilities.
This module provides functionality to generate text while tracking and modifying logits
for watermarking purposes.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from multilingual_watermarking.logit_modification import LogitModificationTracker, LogitModifier
from multilingual_watermarking.paths import Paths
from multilingual_watermarking.utils.embedding_and_paraphrase import evaluate_and_save_paraphrases

# Collection of prompts for text generation
PROMPTS = [
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

# Model configuration
MODEL_NAME = "speakleash/Bielik-4.5B-v3.0-Instruct"
BEGINNING_PROMPT = "Odpowiadaj krótko, precyzyjnie i wyłącznie w języku polskim. "


def setup_model_and_tokenizer(
    model_name: str = MODEL_NAME, device: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    """
    Initialize and setup the model and tokenizer.

    Args:
        model_name: Name of the model to load
        device: Device to run the model on (if None, will use CUDA if available)

    Returns:
        Tuple containing:
        - model: The loaded language model
        - tokenizer: The tokenizer
        - device: The device being used
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer, device


def generate_text(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    tracker: LogitModificationTracker,
    max_length: int = 1000,
    watermark_type: str = "multiple_logits",
) -> str:
    """
    Generate text from the model while tracking logits.

    Args:
        prompt: The input prompt
        model: The language model
        tokenizer: The tokenizer
        device: The device to run on
        tracker: The logit modification tracker
        max_length: Maximum length of generated text

    Returns:
        Generated text as string
    """
    # Tokenize the prompt
    encoded_prompt = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded_prompt["input_ids"].to(device)
    attention_mask = encoded_prompt["attention_mask"].to(device)

    # Generate tokens
    generated = input_ids
    model.eval()

    logit_modifier = LogitModifier(tokenizer=tokenizer)
    with torch.no_grad():
        for pos in range(max_length):
            generated, attention_mask, next_token_id, modified_logits = (
                logit_modifier.generate_output_with_logits(
                    model,
                    device,
                    generated,
                    attention_mask,
                    tracker,
                    position=pos,
                    watermark_type=watermark_type,  # Pass watermark type here
                )
            )

            # Print progress
            if pos % 100 == 0:
                output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                print(f"\nGenerated text so far: {output_text}")

                # Print token information
                token_info = tracker.token_history[-1]
                print(f"Token ID: {token_info['token_id']}")
                print(f"Original logit: {token_info['original_logits'].max().item():.4f}")
                print(f"Modified logit: {token_info['modified_logits'].max().item():.4f}")

            # Stop if EOS token is generated
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def save_output(text: str, file_path: Path) -> None:
    """
    Save generated text to a file.

    Args:
        text: The text to save
        file_path: Path where to save the text
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Output saved to {file_path}")


def main():
    """Main function to run the text generation process."""
    print(f"Loading model: {MODEL_NAME}")

    # Setup model and tokenizer
    paths = Paths()
    model, tokenizer, device = setup_model_and_tokenizer()

    # List of watermark types to iterate over
    watermark_types = [
        "none",  # No watermark (unmasked)
        "random_50",
        "adj_adv_90",
        "feminine_90",
        "verb_comp_90",
    ]

    for prompt_idx, prompt in enumerate(PROMPTS):
        if prompt_idx >= 5:
            return  # Limit to first 5 prompts for testing
        prompt_text = BEGINNING_PROMPT + prompt
        outputs = {}
        for watermark_type in watermark_types:
            print(
                f"\n=== Generating for prompt {prompt_idx} with watermark type: {watermark_type} ==="
            )
            tracker = LogitModificationTracker()
            output_text = generate_text(
                prompt_text,
                model,
                tokenizer,
                device,
                tracker,
                max_length=300,
                watermark_type=watermark_type,
            )
            # Save results
            output_file = paths.GENERATED_TEXT_DIR / f"output_{prompt_idx}_{watermark_type}.txt"
            save_output(output_text, output_file)
            tracker.save_history_to_csv(f"{prompt_idx}_{watermark_type}")
            outputs[watermark_type] = output_text
        evaluate_and_save_paraphrases(
            prompt=prompt,
            prompt_text=prompt_text,
            outputs=outputs,
            watermark_types=watermark_types,
            model=model,
            tokenizer=tokenizer,
            device=device,
            tracker=tracker,
            prompt_idx=prompt_idx,
        )


if __name__ == "__main__":
    main()
