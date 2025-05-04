import torch
from transformers import LogitsProcessor, PreTrainedTokenizer


class ForceStringLogitsProcessor(LogitsProcessor):
    """
    Forces the generation of a specific target string by modifying logits.

    Args:
        target_string (str): The string to force generate.
        tokenizer (PreTrainedTokenizer): The tokenizer used with the model.
        start_index (int): The generation step index at which to start forcing
                           the string (0 means start immediately after the prompt).
    """

    def __init__(self, target_string: str, tokenizer: PreTrainedTokenizer, start_index: int = 0):
        if not isinstance(target_string, str) or len(target_string) == 0:
            raise ValueError("`target_string` must be a non-empty string.")
        if not isinstance(start_index, int) or start_index < 0:
            raise ValueError("`start_index` must be a non-negative integer.")

        self.tokenizer = tokenizer
        self.start_index = start_index
        self.is_forcing = True  # Flag to indicate if we are still forcing

        # Tokenize the target string *without* special tokens
        # We only want the raw token IDs for the string itself.
        self.target_token_ids = self.tokenizer.encode(
            target_string,
            add_special_tokens=False,
            return_tensors=None,  # Ensure it returns a list of ints
        )

        if not self.target_token_ids:
            raise ValueError(f"Target string '{target_string}' resulted in empty token list.")

        self.target_length = len(self.target_token_ids)
        print("Initialized ForceStringLogitsProcessor:")
        print(f"  Target String: '{target_string}'")
        print(f"  Target Token IDs: {self.target_token_ids}")
        print(f"  Target Length: {self.target_length}")
        print(f"  Start Index: {self.start_index}")

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Modifies the logits `scores` to force the next token from the target string.

        Args:
            input_ids (torch.LongTensor): Indices of input sequence tokens in the vocabulary.
                                          Shape: (batch_size, sequence_length)
            scores (torch.FloatTensor): Prediction scores of the language modeling head
                                        (logits for the next token).
                                        Shape: (batch_size, config.vocab_size)

        Returns:
            torch.FloatTensor: The modified logits.
        """
        # input_ids shape is (batch_size, sequence_length)
        # sequence_length includes the prompt + already generated tokens
        # We care about the number of *generated* tokens
        current_generation_step = (
            input_ids.shape[1] - 1
        )  # 0-based index of the token we are *about* to generate

        # Calculate the index within our target string
        force_step_index = current_generation_step - self.start_index

        # Check if we are within the forcing range
        if self.is_forcing and force_step_index >= 0 and force_step_index < self.target_length:
            target_token_id = self.target_token_ids[force_step_index]

            # --- Logit Modification ---
            # Directly modify scores to force the target token ID
            scores.fill_(-float("inf"))
            scores[:, target_token_id] = (
                0.0  # Or some other finite value like scores[:, target_token_id].item() if needed
            )

            # Check if this was the last token of the target string
            if force_step_index == self.target_length - 1:
                print(
                    f"Finished forcing target string at generation step {current_generation_step}."
                )
                self.is_forcing = False  # Stop forcing for subsequent steps

        # If not forcing (before start_index or after target_length), return original scores
        return scores
