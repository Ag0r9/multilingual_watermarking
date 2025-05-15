import unittest
from transformers import AutoTokenizer
from HashForceWatermarkDetector import HashForceWatermarkDetector


class TestHashForceWatermarkDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        cls.detector = HashForceWatermarkDetector(cls.tokenizer)
        cls.secret_key = "test_secret_key"

    def test_watermark_generation(self):
        # Test basic watermark generation
        text = "This is a test text"
        watermark = self.detector.generate_watermark(text)
        self.assertEqual(len(watermark), 32)
        self.assertTrue(all(c in '0123456789abcdef' for c in watermark))

        # Test with secret key
        watermark_with_key = self.detector.generate_watermark(text, self.secret_key)
        self.assertNotEqual(watermark, watermark_with_key)

        # Test deterministic generation
        watermark2 = self.detector.generate_watermark(text)
        self.assertEqual(watermark, watermark2)

    def test_watermark_detection(self):
        # Generate a watermarked text
        base_text = "This is a test text"
        watermark = self.detector.generate_watermark(base_text)
        watermarked_text = f"This is {watermark} a test text"

        # Test detection
        results = self.detector.detect_watermark(watermarked_text)
        self.assertTrue(results["is_watermarked"])
        self.assertEqual(len(results["hash_positions"]), 1)
        self.assertEqual(results["hash_positions"][0][0], watermark)

        # Test with secret key
        watermark_with_key = self.detector.generate_watermark(base_text, self.secret_key)
        watermarked_text_with_key = f"This is {watermark_with_key} a test text"
        results = self.detector.detect_watermark(watermarked_text_with_key, self.secret_key)
        self.assertTrue(results["is_watermarked"])

        # Test with no watermark
        clean_text = "This is a clean text"
        results = self.detector.detect_watermark(clean_text)
        self.assertFalse(results["is_watermarked"])

    def test_text_integrity(self):
        # Create watermarked text
        base_text = "This is a test text"
        watermark = self.detector.generate_watermark(base_text)
        watermarked_text = f"This is {watermark} a test text"

        # Test unmodified text
        results = self.detector.verify_text_integrity(watermarked_text)
        self.assertTrue(results["is_valid"])

        # Test modified text
        modified_text = watermarked_text.replace("test", "modified")
        results = self.detector.verify_text_integrity(modified_text)
        self.assertFalse(results["is_valid"])

        # Test with secret key
        watermark_with_key = self.detector.generate_watermark(base_text, self.secret_key)
        watermarked_text_with_key = f"This is {watermark_with_key} a test text"
        results = self.detector.verify_text_integrity(watermarked_text_with_key, self.secret_key)
        self.assertTrue(results["is_valid"])

    def test_comprehensive_analysis(self):
        # Create watermarked text
        base_text = "This is a test text"
        watermark = self.detector.generate_watermark(base_text)
        watermarked_text = f"This is {watermark} a test text"

        # Test comprehensive analysis
        results = self.detector.analyze_text(watermarked_text)
        self.assertTrue(results["watermark_detected"])
        self.assertTrue(results["integrity_verified"])
        self.assertTrue(len(results["found_hashes"]) > 0)
        self.assertIsNotNone(results["verification_details"])
        self.assertIsNotNone(results["integrity_check"])

        # Test with modified text
        modified_text = watermarked_text.replace("test", "modified")
        results = self.detector.analyze_text(modified_text)
        self.assertFalse(results["integrity_verified"])

        # Test with clean text
        clean_text = "This is a clean text"
        results = self.detector.analyze_text(clean_text)
        self.assertFalse(results["watermark_detected"])
        self.assertFalse(results["integrity_verified"])


if __name__ == '__main__':
    unittest.main() 