from typing import Tuple, Dict, Callable
import random
import pandas as pd
import torch
import torch.nn.functional as F
import spacy
from pathlib import Path

from multilingual_watermarking.paths import Paths, get_timestamp


class LogitModificationTracker:
    def __init__(self):
        self.paths = Paths()
        self.token_history = []  # List to store token generation history

    def add_token_info(
        self,
        token_id: int,
        original_logits: torch.Tensor,
        modified_logits: torch.Tensor,
        position: int,
    ):
        self.token_history.append(
            {
                "token_id": token_id,
                "original_logits": original_logits.detach().cpu(),
                "modified_logits": modified_logits.detach().cpu(),
                "position": position,
            }
        )

    def get_history(self):
        return self.token_history

    def save_history_to_csv(self, watermark_type: str = "unknown"):
        history = [
            {
                "position": entry["position"],
                "token_id": entry["token_id"],
                "original_logits": entry["original_logits"].max().item(),
                "modified_logits": entry["modified_logits"].max().item(),
            }
            for entry in self.token_history
        ]
        df = pd.DataFrame(history)
        timestamp = get_timestamp()
        # Ensure the directory exists
        self.paths.LOGITS_DIR.mkdir(parents=True, exist_ok=True)
        # Save to CSV with watermark_type in filename
        df.to_csv(self.paths.LOGITS_DIR / f"logits_{watermark_type}_{timestamp}.csv", index=False)


class LogitModifier:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.nlp = spacy.load("pl_core_news_lg")  # Load spaCy for POS tagging and gender detection

    def multiple_logits(self, logits: torch.Tensor, multiplication_factor: float = 0.5) -> torch.Tensor:
        """
        Modify logits according to the specified factor.
        Args:
            logits: Original logits
            multiplication_factor: Factor to modify logits (e.g., 0.5 for 50% reduction)
        Returns:
            Modified logits
        """
        return logits * multiplication_factor

    def random_50(self, logits: torch.Tensor, doc: spacy.tokens.Doc, reduction_factor: float = 0.8) -> torch.Tensor:
        """
        Apply random watermark to 50% of tokens.
        Args:
            logits: Original logits
            doc: spaCy document
            reduction_factor: Factor to reduce logits by
        Returns:
            Modified logits
        """
        tokens = self.tokenizer.encode(doc.text)
        mask_indices = random.sample(range(len(tokens)), k=int(len(tokens) * 0.5))
        modified_logits = logits.clone()
        for idx in mask_indices:
            modified_logits[0, idx] *= reduction_factor
        return modified_logits

    def adj_adv_90(self, logits: torch.Tensor, doc: spacy.tokens.Doc, reduction_factor: float = 0.8) -> torch.Tensor:
        """
        Apply watermark to 90% of adjectives and adverbs.
        Args:
            logits: Original logits
            doc: spaCy document
            reduction_factor: Factor to reduce logits by
        Returns:
            Modified logits
        """
        tokens = self.tokenizer.encode(doc.text)
        adj_adv_indices = [
            i for i, token in enumerate(doc)
            if token.pos_ in ["ADJ", "ADV"]
        ]
        if not adj_adv_indices:
            return logits
        mask_indices = random.sample(
            adj_adv_indices,
            k=int(len(adj_adv_indices) * 0.9)
        )
        modified_logits = logits.clone()
        for idx in mask_indices:
            modified_logits[0, idx] *= reduction_factor
        return modified_logits

    def feminine_90(self, logits: torch.Tensor, doc: spacy.tokens.Doc, reduction_factor: float = 0.8) -> torch.Tensor:
        """
        Apply watermark to 90% of feminine gender words.
        Args:
            logits: Original logits
            doc: spaCy document
            reduction_factor: Factor to reduce logits by
        Returns:
            Modified logits
        """
        tokens = self.tokenizer.encode(doc.text)
        feminine_indices = [
            i for i, token in enumerate(doc)
            if token.morph.get("Gender") == ["Fem"]
        ]
        if not feminine_indices:
            return logits
        mask_indices = random.sample(
            feminine_indices,
            k=int(len(feminine_indices) * 0.9)
        )
        modified_logits = logits.clone()
        for idx in mask_indices:
            modified_logits[0, idx] *= reduction_factor
        return modified_logits

    def verb_comp_90(self, logits: torch.Tensor, doc: spacy.tokens.Doc, reduction_factor: float = 0.8) -> torch.Tensor:
        """
        Apply watermark to 90% of verbs and complements.
        Args:
            logits: Original logits
            doc: spaCy document
            reduction_factor: Factor to reduce logits by
        Returns:
            Modified logits
        """
        tokens = self.tokenizer.encode(doc.text)
        verb_comp_indices = [
            i for i, token in enumerate(doc)
            if token.pos_ in ["VERB", "AUX"] or token.dep_ in ["ccomp", "xcomp"]
        ]
        if not verb_comp_indices:
            return logits
        mask_indices = random.sample(
            verb_comp_indices,
            k=int(len(verb_comp_indices) * 0.9)
        )
        modified_logits = logits.clone()
        for idx in mask_indices:
            modified_logits[0, idx] *= reduction_factor
        return modified_logits

    def generate_output_with_logits(
        self,
        model,
        device: torch.device,
        generated: torch.Tensor,
        attention_mask: torch.Tensor,
        tracker: LogitModificationTracker,
        position: int,
        watermark_type: str = "multiple_logits",
        reduction_factor: float = 0.8
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """
        Generate output from the model and track both original and modified logits.

        Args:
            model: The language model
            device: Device to run on
            generated: Generated tokens so far
            attention_mask: Attention mask for the generated tokens
            tracker: Logit modification tracker
            position: Current position in generation
            watermark_type: Type of watermark to apply
            reduction_factor: Factor to reduce logits by

        Returns:
            Tuple containing:
            - generated: Updated generated tokens
            - attention_mask: Updated attention mask
            - next_token_id: Next token ID
            - modified_logits: Modified logits
        """
        watermark_functions: Dict[str, Callable] = {
            "multiple_logits": lambda l, d: self.multiple_logits(l, reduction_factor),
            "random_50": lambda l, d: self.random_50(l, d, reduction_factor),
            "adj_adv_90": lambda l, d: self.adj_adv_90(l, d, reduction_factor),
            "feminine_90": lambda l, d: self.feminine_90(l, d, reduction_factor),
            "verb_comp_90": lambda l, d: self.verb_comp_90(l, d, reduction_factor),
        }

        outputs = model(input_ids=generated, attention_mask=attention_mask)
        original_logits = outputs.logits[:, -1, :]

        current_text = self.tokenizer.decode(generated[0])
        doc = self.nlp(current_text)

        watermark_func = watermark_functions.get(watermark_type, watermark_functions["multiple_logits"])
        modified_logits = watermark_func(original_logits, doc)

        next_token_id = torch.argmax(modified_logits, dim=-1).unsqueeze(-1)

        tracker.add_token_info(
            token_id=next_token_id.item(),
            original_logits=original_logits,
            modified_logits=modified_logits,
            position=position,
        )

        generated = torch.cat([generated, next_token_id], dim=-1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype),
            ],
            dim=1,
        )

        return generated, attention_mask, next_token_id, modified_logits
