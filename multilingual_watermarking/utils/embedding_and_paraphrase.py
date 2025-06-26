import os
from typing import Any, Dict, List

import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OAI_API_KEY"))


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
        response = client.chat.completions.create(model=model,
        messages=[{"role": "user", "content": f"Paraphrase this text: {text}"}])
        paraphrases.append(response.choices[0].message.content)
    return paraphrases
    # raise NotImplementedError("Implement paraphrasing using your chosen LLM API.")


def compute_logprob(
    text: str,
    prompt: str,
    model,
    tokenizer,
    device,
    tracker,
    watermark_type: str
) -> float:
    """
    Compute the log-probability of generating the given text by the watermarked model.
    """
    # This can be implemented similarly to calculate_sequence_probability in check_logits.py
    # (copy or import and adapt as needed)
    raise NotImplementedError("Implement log-probability calculation for the watermarked model.")


def save_to_csv(data: Dict[str, Any], filename: str = "paraphrase_eval.csv"):
    """
    Save the evaluation results to a CSV file for manual annotation.
    """
    import pandas as pd
    df = pd.DataFrame([data])
    df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False) 