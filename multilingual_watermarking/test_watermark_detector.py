import unittest
from transformers import AutoTokenizer
from WatermarkDetector import WatermarkDetector


class TestWatermarkDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize tokenizer and detector once for all tests
        cls.tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Using GPT-2 tokenizer for testing
        cls.detector = WatermarkDetector(cls.tokenizer)

    def test_forced_string_detection(self):
        # Test case 1: Watermark at the beginning
        text = "WATERMARK This is a test sentence."
        target = "WATERMARK"
        detected, positions = self.detector.detect_forced_string(text, target, start_index=0)
        self.assertTrue(detected)
        self.assertTrue(len(positions) > 0)

        # Test case 2: Watermark in the middle
        text = "This is a WATERMARK test sentence."
        detected, positions = self.detector.detect_forced_string(text, target, start_index=2)
        self.assertTrue(detected)
        self.assertTrue(len(positions) > 0)

        # Test case 3: No watermark present
        text = "This is a clean text without watermark."
        detected, positions = self.detector.detect_forced_string(text, target, start_index=0)
        self.assertFalse(detected)
        self.assertEqual(len(positions), 0)

        # Test case 4: Text too short for watermark
        text = "Short"
        detected, positions = self.detector.detect_forced_string(text, "LongWatermark", start_index=0)
        self.assertFalse(detected)
        self.assertEqual(len(positions), 0)

    def test_hash_based_watermark_detection(self):
        # Test case 1: Valid hash in text
        hash_value = "a1b2c3d4e5f6"
        text = f"This text contains a hash {hash_value} in the middle."
        results = self.detector.detect_hash_based_watermark(text, hash_length=12)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], hash_value)

        # Test case 2: Multiple hashes
        text = f"Hash1: {hash_value} and Hash2: {hash_value}"
        results = self.detector.detect_hash_based_watermark(text, hash_length=12)
        self.assertEqual(len(results), 2)

        # Test case 3: No hash present
        text = "This text contains no hash values."
        results = self.detector.detect_hash_based_watermark(text)
        self.assertEqual(len(results), 0)

        # Test case 4: Invalid hex characters
        text = "This contains invalid hash: g1h2i3j4k5l6"
        results = self.detector.detect_hash_based_watermark(text, hash_length=12)
        self.assertEqual(len(results), 0)

    def test_hash_watermark_generation(self):
        # Test case 1: Basic hash generation
        seed = "test_seed"
        hash_value = self.detector.generate_hash_watermark(seed, length=32)
        self.assertEqual(len(hash_value), 32)
        self.assertTrue(all(c in '0123456789abcdef' for c in hash_value))

        # Test case 2: Different seeds produce different hashes
        hash1 = self.detector.generate_hash_watermark("seed1")
        hash2 = self.detector.generate_hash_watermark("seed2")
        self.assertNotEqual(hash1, hash2)

        # Test case 3: Same seed produces same hash
        hash3 = self.detector.generate_hash_watermark("seed1")
        self.assertEqual(hash1, hash3)

        # Test case 4: Custom length
        hash_value = self.detector.generate_hash_watermark(seed, length=16)
        self.assertEqual(len(hash_value), 16)

    def test_comprehensive_analysis(self):
        # Create a text with both types of watermarks
        hash_value = self.detector.generate_hash_watermark("test_seed", length=16)
        forced_string = "WATERMARK"
        text = f"This {forced_string} text contains a hash {hash_value} value."
        
        # Test with known watermark
        results = self.detector.analyze_text(text, known_watermarks=[forced_string])
        
        # Check forced string detection
        self.assertTrue(len(results["forced_strings"]) > 0)
        self.assertEqual(results["forced_strings"][0]["watermark"], forced_string)
        
        # Check hash detection
        self.assertTrue(len(results["hash_watermarks"]) > 0)
        self.assertEqual(results["hash_watermarks"][0][0], hash_value)
        
        # Check analysis string
        self.assertEqual(results["analysis"], "Watermarks detected in text")

        # Test clean text
        clean_text = "This is a clean text without any watermarks."
        clean_results = self.detector.analyze_text(clean_text)
        self.assertEqual(clean_results["analysis"], "No known watermarks detected")


if __name__ == '__main__':
    unittest.main() 