from typing import List, Optional, Tuple
import hashlib
from transformers import PreTrainedTokenizer


class WatermarkDetector:
    """
    Detector for identifying watermarks in generated text.
    Currently supports detection of forced string watermarks.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Initialize the watermark detector.

        Args:
            tokenizer (PreTrainedTokenizer): The same tokenizer used during text generation
        """
        self.tokenizer = tokenizer

    def detect_forced_string(
        self, text: str, target_string: str, start_index: int = 0
    ) -> Tuple[bool, List[int]]:
        """
        Detect if a specific forced string watermark is present in the text.

        Args:
            text (str): The text to analyze for watermarks
            target_string (str): The string that was used as watermark
            start_index (int): The token position where the watermark was supposed to start

        Returns:
            Tuple[bool, List[int]]: A tuple containing:
                - bool: Whether the watermark was detected
                - List[int]: Positions where the watermark was found (token indices)
        """
        # Tokenize both the full text and the target string
        text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        target_tokens = self.tokenizer.encode(target_string, add_special_tokens=False)

        # If text is shorter than start_index + target length, watermark can't be present
        if len(text_tokens) < start_index + len(target_tokens):
            return False, []

        # Check if the target tokens appear at the expected position
        expected_slice = text_tokens[start_index:start_index + len(target_tokens)]
        if expected_slice == target_tokens:
            return True, list(range(start_index, start_index + len(target_tokens)))

        return False, []

    def detect_hash_based_watermark(
        self, text: str, hash_length: int = 32, window_size: Optional[int] = None
    ) -> List[Tuple[str, int]]:
        """
        Detect potential hash-based watermarks in the text by looking for hex-like sequences.
        This is useful for detecting watermarks that use hash values embedded in the text.

        Args:
            text (str): The text to analyze for hash-based watermarks
            hash_length (int): Expected length of the hash string
            window_size (Optional[int]): Size of the sliding window to use for detection.
                                       If None, will use the hash_length.

        Returns:
            List[Tuple[str, int]]: List of (potential_hash, position) tuples where hashes were found
        """
        if window_size is None:
            window_size = hash_length

        found_hashes = []
        words = text.split()
        
        for i, word in enumerate(words):
            # Check if the word could be a hash (hex characters of correct length)
            if len(word) == hash_length and all(c in '0123456789abcdefABCDEF' for c in word):
                found_hashes.append((word, i))

        return found_hashes

    @staticmethod
    def generate_hash_watermark(seed: str, length: int = 32) -> str:
        """
        Generate a hash-based watermark from a seed string.
        This can be used to verify if a detected hash matches an expected watermark.

        Args:
            seed (str): Seed string to generate the hash from
            length (int): Desired length of the hash string

        Returns:
            str: Generated hash string
        """
        hash_obj = hashlib.sha256(seed.encode())
        return hash_obj.hexdigest()[:length]

    def analyze_text(
        self, text: str, known_watermarks: Optional[List[str]] = None
    ) -> dict:
        """
        Perform comprehensive watermark analysis on the text.
        This method combines different detection strategies.

        Args:
            text (str): The text to analyze
            known_watermarks (Optional[List[str]]): List of known watermark strings to check for

        Returns:
            dict: Analysis results containing detected watermarks and their positions
        """
        results = {
            "forced_strings": [],
            "hash_watermarks": [],
            "analysis": ""
        }

        # Check for known forced string watermarks
        if known_watermarks:
            for watermark in known_watermarks:
                detected, positions = self.detect_forced_string(text, watermark)
                if detected:
                    results["forced_strings"].append({
                        "watermark": watermark,
                        "positions": positions
                    })

        # Look for potential hash-based watermarks
        hash_candidates = self.detect_hash_based_watermark(text)
        if hash_candidates:
            results["hash_watermarks"] = hash_candidates

        # Generate analysis summary
        if results["forced_strings"] or results["hash_watermarks"]:
            results["analysis"] = "Watermarks detected in text"
        else:
            results["analysis"] = "No known watermarks detected"

        return results 