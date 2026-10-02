import unittest
from core.translate import chunk_text_safely, translate_text

class TestTranslate(unittest.TestCase):
    def test_chunk_text_safely_empty(self):
        self.assertEqual(chunk_text_safely(""), [])
        self.assertEqual(chunk_text_safely("   "), [])

    def test_chunk_text_safely_sentence_boundaries(self):
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = chunk_text_safely(text, max_chars=45)
        self.assertGreater(len(chunks), 1)
        # Ensure no words are cut in half
        for c in chunks:
            self.assertFalse(c.endswith("sentenc"))
            self.assertTrue(len(c) <= 45)

    def test_translate_text_empty(self):
        self.assertEqual(translate_text(""), "")

if __name__ == "__main__":
    unittest.main()
