from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class WatermarkResult:
    original_text: str
    watermarked_text: str
    cosine_similarity: float
    paraphrases: List[Dict[str, float]]  # List of dicts with text and probability
    grammatical_correctness: bool


class WatermarkEvaluator:
    def __init__(
        self,
        model_name: str,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        spacy_model: str = "pl_core_news_lg",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        # Load the main model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(self.device)

        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name).to(self.device)

        # Load spaCy for POS tagging and gender detection
        self.nlp = spacy.load(spacy_model)

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text using the sentence transformer model."""
        with torch.no_grad():
            embedding = self.embedding_model.encode(text, convert_to_tensor=True)
        return embedding.cpu().numpy()

    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def generate_paraphrases(self, text: str, num_paraphrases: int = 3) -> List[str]:
        """Generate paraphrases using the main model."""
        paraphrases = []
        prompt_template = "Przepisz poniższy tekst innymi słowami, zachowując znaczenie: {text}"

        for _ in range(num_paraphrases):
            prompt = prompt_template.format(text=text)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                )

            paraphrase = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            paraphrases.append(paraphrase)

        return paraphrases

    def calculate_sequence_probability(self, text: str) -> float:
        """Calculate the probability of generating a sequence using the watermarked model."""
        tokens = self.tokenizer.encode(text)
        total_log_prob = 0.0

        for i, target_token in enumerate(tokens):
            input_tokens = tokens[:i]
            encoded = self.tokenizer(
                self.tokenizer.decode(input_tokens),
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)

            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits, dim=-1)
            total_log_prob += torch.log(probs[target_token]).item()

        return float(total_log_prob)

    def apply_watermark(
        self, text: str, watermark_type: str, reduction_factor: float = 0.8
    ) -> str:
        """Apply watermark to text based on specified type."""
        doc = self.nlp(text)
        tokens = self.tokenizer.encode(text)
        modified_tokens = tokens.copy()

        if watermark_type == "random_50":
            # Random mask on 50% of tokens
            mask_indices = np.random.choice(
                len(tokens), size=int(len(tokens) * 0.5), replace=False
            )
            for idx in mask_indices:
                modified_tokens[idx] = int(modified_tokens[idx] * reduction_factor)

        elif watermark_type == "adj_adv_90":
            # Mask 90% of adjectives and adverbs
            pos_tags = [token.pos_ for token in doc]
            adj_adv_indices = [i for i, pos in enumerate(pos_tags) if pos in ["ADJ", "ADV"]]
            mask_indices = np.random.choice(
                adj_adv_indices, size=int(len(adj_adv_indices) * 0.9), replace=False
            )
            for idx in mask_indices:
                modified_tokens[idx] = int(modified_tokens[idx] * reduction_factor)

        elif watermark_type == "feminine_90":
            # Mask 90% of feminine gender words
            feminine_indices = [
                i for i, token in enumerate(doc) if token.morph.get("Gender") == ["Fem"]
            ]
            mask_indices = np.random.choice(
                feminine_indices, size=int(len(feminine_indices) * 0.9), replace=False
            )
            for idx in mask_indices:
                modified_tokens[idx] = int(modified_tokens[idx] * reduction_factor)

        elif watermark_type == "verb_comp_90":
            # Mask 90% of verbs and complements
            verb_comp_indices = [
                i
                for i, token in enumerate(doc)
                if token.pos_ in ["VERB", "AUX"] or token.dep_ in ["ccomp", "xcomp"]
            ]
            mask_indices = np.random.choice(
                verb_comp_indices, size=int(len(verb_comp_indices) * 0.9), replace=False
            )
            for idx in mask_indices:
                modified_tokens[idx] = int(modified_tokens[idx] * reduction_factor)

        return self.tokenizer.decode(modified_tokens)

    def evaluate_watermark(
        self, prompt: str, watermark_type: str, num_paraphrases: int = 3
    ) -> WatermarkResult:
        """Evaluate watermark effectiveness for a given prompt."""
        # Generate original and watermarked text
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=512, num_return_sequences=1)
        original_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        watermarked_text = self.apply_watermark(original_text, watermark_type)

        # Calculate similarity
        similarity = self.calculate_cosine_similarity(original_text, watermarked_text)

        # Generate and evaluate paraphrases
        paraphrases = self.generate_paraphrases(watermarked_text, num_paraphrases)
        paraphrase_results = []

        for paraphrase in paraphrases:
            prob = self.calculate_sequence_probability(paraphrase)
            paraphrase_results.append({"text": paraphrase, "probability": prob})

        # Check grammatical correctness (simple heuristic)
        doc = self.nlp(watermarked_text)
        grammatical_correctness = all(not token.is_punct or token.is_space for token in doc)

        return WatermarkResult(
            original_text=original_text,
            watermarked_text=watermarked_text,
            cosine_similarity=similarity,
            paraphrases=paraphrase_results,
            grammatical_correctness=grammatical_correctness,
        )

    def run_evaluation(
        self, prompts: List[str], watermark_types: List[str], output_dir: Path
    ) -> None:
        """Run full evaluation on multiple prompts and watermark types."""
        results = []

        for prompt in prompts:
            for watermark_type in watermark_types:
                result = self.evaluate_watermark(prompt, watermark_type)
                results.append(
                    {
                        "prompt": prompt,
                        "watermark_type": watermark_type,
                        "original_text": result.original_text,
                        "watermarked_text": result.watermarked_text,
                        "cosine_similarity": result.cosine_similarity,
                        "paraphrases": result.paraphrases,
                        "grammatical_correctness": result.grammatical_correctness,
                    }
                )

        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "watermark_evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
