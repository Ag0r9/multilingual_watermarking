from typing import Tuple

import pandas as pd
import torch
import torch.nn.functional as F

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

    def save_history_to_csv(self):
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
        # Save to CSV
        df.to_csv(self.paths.LOGITS_DIR / f"logits_{timestamp}.csv", index=False)


def multiple_logits(logits: torch.Tensor, multiplication_factor: float = 0.5) -> torch.Tensor:
    """
    Modify logits according to the specified factor.
    Args:
        logits: Original logits
        multiplication_factor: Factor to modify logits (e.g., 0.5 for 50% reduction)
    Returns:
        Modified logits
    """
    return logits * multiplication_factor


def generate_output_with_logits(
    model,
    device: torch.device,
    generated: torch.Tensor,
    attention_mask: torch.Tensor,
    tracker: LogitModificationTracker,
    position: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """
    Generate output from the model and track both original and modified logits.
    """
    outputs = model(input_ids=generated, attention_mask=attention_mask)
    original_logits = outputs.logits[:, -1, :]

    # Modify logits
    modified_logits = multiple_logits(original_logits, 0.5)

    # Get next token based on modified logits
    next_token_id = torch.argmax(modified_logits, dim=-1).unsqueeze(-1)

    # Track the token and its logits
    tracker.add_token_info(
        token_id=next_token_id.item(),
        original_logits=original_logits,
        modified_logits=modified_logits,
        position=position,
    )

    # Update generated sequence and attention mask
    generated = torch.cat([generated, next_token_id], dim=-1)
    attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype),
        ],
        dim=1,
    )

    return generated, attention_mask, next_token_id, modified_logits
