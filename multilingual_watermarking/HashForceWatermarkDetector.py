from typing import List, Optional, Tuple, Dict
import hashlib
from transformers import PreTrainedTokenizer


class HashForceWatermarkDetector:
    """
    Specialized detector for watermarks created by ForceStringLogitsProcessor
    that include hash-based information in the generated text.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, hash_length: int = 32):
        """
        Initialize the watermark detector.

        Args:
            tokenizer (PreTrainedTokenizer): The same tokenizer used during text generation
            hash_length (int): Length of hash strings to look for (default: 32 for SHA-256)
        """
        self.tokenizer = tokenizer
        self.hash_length = hash_length

    def generate_watermark(self, text: str, secret_key: str = "") -> str:
        """
        Generate a hash-based watermark from input text and optional secret key.
        This should match the hash generation in the ForceStringLogitsProcessor.

        Args:
            text (str): Input text to generate watermark from
            secret_key (str): Optional secret key to make watermark harder to forge

        Returns:
            str: Generated hash watermark
        """
        combined = text + secret_key
        hash_obj = hashlib.sha256(combined.encode())
        return hash_obj.hexdigest()[:self.hash_length]

    def detect_watermark(
        self, text: str, secret_key: str = "", start_index: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Detect if text contains a valid watermark.
        
        Args:
            text (str): Text to analyze for watermarks
            secret_key (str): Optional secret key used in watermark generation
            start_index (Optional[int]): Expected position of the watermark in tokens

        Returns:
            Dict containing:
                - is_watermarked (bool): Whether a valid watermark was found
                - hash_positions (List[Tuple[str, int]]): List of (hash, position) tuples
                - verification (Dict): Verification results if watermark was found
        """
        # Tokenize the text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        words = text.split()
        
        # Find all potential hash strings
        hash_positions = []
        for i, word in enumerate(words):
            if (len(word) == self.hash_length and 
                all(c in '0123456789abcdefABCDEF' for c in word)):
                hash_positions.append((word.lower(), i))

        if not hash_positions:
            return {
                "is_watermarked": False,
                "hash_positions": [],
                "verification": None
            }

        # Generate expected watermark
        expected_hash = self.generate_watermark(text, secret_key).lower()
        
        # Verify each found hash
        verification_results = []
        for found_hash, position in hash_positions:
            # Remove the hash from text to verify it was generated from the rest
            words_without_hash = words.copy()
            words_without_hash[position] = ""
            text_without_hash = " ".join(filter(None, words_without_hash))
            
            # Generate verification hash
            verification_hash = self.generate_watermark(text_without_hash, secret_key).lower()
            
            verification_results.append({
                "position": position,
                "found_hash": found_hash,
                "expected_hash": verification_hash,
                "is_valid": found_hash == verification_hash
            })

        # Check if any valid watermarks were found
        is_watermarked = any(v["is_valid"] for v in verification_results)

        return {
            "is_watermarked": is_watermarked,
            "hash_positions": hash_positions,
            "verification": verification_results
        }

    def verify_text_integrity(self, text: str, secret_key: str = "") -> Dict[str, any]:
        """
        Verify if the text has been modified since watermarking.

        Args:
            text (str): Text to verify
            secret_key (str): Optional secret key used in watermark generation

        Returns:
            Dict containing verification results
        """
        detection_results = self.detect_watermark(text, secret_key)
        
        if not detection_results["is_watermarked"]:
            return {
                "is_valid": False,
                "reason": "No watermark found",
                "details": None
            }

        # Check each verification result
        valid_watermarks = [v for v in detection_results["verification"] if v["is_valid"]]
        
        if not valid_watermarks:
            return {
                "is_valid": False,
                "reason": "All found watermarks are invalid",
                "details": detection_results["verification"]
            }

        return {
            "is_valid": True,
            "reason": "Valid watermark(s) found",
            "details": valid_watermarks
        }

    def analyze_text(self, text: str, secret_key: str = "") -> Dict[str, any]:
        """
        Perform comprehensive watermark analysis on the text.

        Args:
            text (str): Text to analyze
            secret_key (str): Optional secret key used in watermark generation

        Returns:
            Dict containing analysis results
        """
        # Get basic watermark detection results
        detection_results = self.detect_watermark(text, secret_key)
        
        # Get integrity verification results
        integrity_results = self.verify_text_integrity(text, secret_key)
        
        # Combine results
        return {
            "watermark_detected": detection_results["is_watermarked"],
            "integrity_verified": integrity_results["is_valid"],
            "found_hashes": detection_results["hash_positions"],
            "verification_details": detection_results["verification"],
            "integrity_check": integrity_results
        } 