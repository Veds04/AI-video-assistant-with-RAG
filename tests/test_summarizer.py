import unittest
from core.summarizer import split_transcript

class TestSummarizer(unittest.TestCase):
    def test_split_transcript_empty(self):
        chunks = split_transcript("")
        self.assertEqual(chunks, [])
        
        chunks = split_transcript("   ")
        self.assertEqual(chunks, [])
        
    def test_split_transcript_small(self):
        chunks = split_transcript("Hello world.")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Hello world.")
        
    def test_split_transcript_large(self):
        text = "Hello world. " * 1000
        chunks = split_transcript(text, max_chars=1000)
        self.assertTrue(len(chunks) > 1)
        self.assertTrue(all(len(c) <= 1000 for c in chunks))

if __name__ == '__main__':
    unittest.main()
