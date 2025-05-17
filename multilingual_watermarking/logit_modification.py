from typing import Tuple

import torch


class LogitModificationTracker:
    def __init__(self):
        self.token_history = []  # List to store token generation history

    def add_token_info(
        self, token_id: int, original_logits: torch.Tensor, modified_logits: torch.Tensor
    ):
        self.token_history.append(
            {
                "token_id": token_id,
                "original_logits": original_logits.detach().cpu(),
                "modified_logits": modified_logits.detach().cpu(),
            }
        )

    def get_history(self):
        return self.token_history


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
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """
    Generate output from the model and track both original and modified logits.
    """
    outputs = model(input_ids=generated, attention_mask=attention_mask)
    original_logits = outputs.logits[:, -1, :]

    # Modify logits
    modified_logits = multiple_logits(original_logits)

    # Get next token based on modified logits
    next_token_id = torch.argmax(modified_logits, dim=-1).unsqueeze(-1)

    # Track the token and its logits
    tracker.add_token_info(
        token_id=next_token_id.item(),
        original_logits=original_logits,
        modified_logits=modified_logits,
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
