from typing import Any, Dict, List

import numpy as np
from openai import OpenAI
import pandas as pd

from multilingual_watermarking.paths import Paths

client = OpenAI()


def get_embedding(text: str, model: str = "text-embedding-ada-002") -> np.ndarray:
    """
    Get embedding for a given text using a large LLM API (e.g., OpenAI, Gemini).
    """
    # Example using OpenAI (pseudo-code):
    response = client.embeddings.create(input=text, model=model)
    return np.array(response.data[0].embedding)
    # raise NotImplementedError("Implement embedding retrieval using your chosen LLM API.")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def paraphrase_text(text: str, n: int = 5, model: str = "gpt-3.5-turbo") -> List[str]:
    """
    Generate n paraphrases of the input text using a large LLM API.
    """
    # Example using OpenAI Chat API (pseudo-code):
    paraphrases = []
    for _ in range(n):
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": f"Paraphrase this text: {text}"}]
        )
        paraphrases.append(response.choices[0].message.content)
    return paraphrases
    # raise NotImplementedError("Implement paraphrasing using your chosen LLM API.")


def compute_logprob(
    text: str, prompt: str, model, tokenizer, device, tracker, watermark_type: str
) -> float:
    """
    Compute the log-probability of generating the given text by the watermarked model.
    """
    # This can be implemented similarly to calculate_sequence_probability in check_logits.py
    # (copy or import and adapt as needed)
    raise NotImplementedError("Implement log-probability calculation for the watermarked model.")


def save_parahprase_results_to_csv(data: Dict[str, Any], filename: str = "paraphrase_eval.csv"):
    """
    Save the evaluation results to a CSV file for manual annotation.
    """
    paths: Paths = Paths()
    if not paths.RESULTS_DIR.exists():
        paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = paths.RESULTS_DIR / filename
    df = pd.DataFrame([data])
    df.to_csv(filename, mode="a", header=not pd.io.common.file_exists(filename), index=False)


def evaluate_and_save_paraphrases(
    prompt: str,
    prompt_text: str,
    outputs: dict,
    watermark_types: list,
    model,
    tokenizer,
    device,
    tracker,
    prompt_idx: int,
):
    """
    Evaluate paraphrases and similarities for masked outputs and save results to CSV.
    """
    unmasked_text = outputs["none"]
    for masked_type in watermark_types:
        if masked_type == "none":
            continue
        masked_text = outputs[masked_type]
        try:
            unmasked_emb = get_embedding(unmasked_text)
            masked_emb = get_embedding(masked_text)
            similarity = cosine_similarity(unmasked_emb, masked_emb)
        except NotImplementedError:
            similarity = None
        try:
            paraphrases = paraphrase_text(masked_text, n=5)
        except NotImplementedError:
            paraphrases = []
        paraphrase_results = []
        for para in paraphrases:
            try:
                para_emb = get_embedding(para)
                para_similarity = cosine_similarity(unmasked_emb, para_emb)
            except NotImplementedError:
                para_similarity = None
            try:
                para_prob = compute_logprob(
                    para, prompt_text, model, tokenizer, device, tracker, masked_type
                )
            except NotImplementedError:
                para_prob = None
            paraphrase_results.append(
                {"paraphrase": para, "similarity": para_similarity, "probability": para_prob}
            )
        save_parahprase_results_to_csv(
            {
                "prompt": prompt,
                "unmasked": unmasked_text,
                "masked": masked_text,
                "similarity": similarity,
                "paraphrases": paraphrase_results,
                "watermark_type": masked_type,
                "prompt_idx": prompt_idx,
            }
        )
