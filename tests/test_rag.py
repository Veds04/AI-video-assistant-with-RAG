import os
import shutil
import tempfile
import unittest
from dotenv import load_dotenv

from core.rag_engine import VideoRAG, format_docs
from core.vector_store import create_transcript_chunks, build_vector_store

load_dotenv()

SAMPLE_TRANSCRIPT = """
The team discussed the launch of the AI dashboard.

Rahul will finish the frontend by Friday.

Priya will prepare the database schema by Wednesday.

The team decided to use PostgreSQL for the production database.

The initial launch is planned for October 15.

The deployment budget has not been finalized yet.
"""

class TestRAGEngine(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.rag = VideoRAG.from_transcript(
            SAMPLE_TRANSCRIPT,
            video_id="sample_test_video",
            persist_directory=self.test_dir,
            k=2,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_empty_question(self):
        result = self.rag.ask_with_sources("")
        self.assertEqual(result["answer"], "Please provide a valid question.")
        self.assertEqual(result["sources"], [])

    def test_retrieval_relevance(self):
        docs = self.rag.retrieve_context("Who is responsible for the frontend?")
        self.assertGreater(len(docs), 0)
        content = " ".join([d.page_content for d in docs])
        self.assertIn("Rahul", content)

    def test_format_docs(self):
        docs = self.rag.retrieve_context("database")
        formatted = format_docs(docs)
        self.assertIn("[Chunk", formatted)
        self.assertIn("PostgreSQL", formatted)

    def test_llm_e2e_rag(self):
        """Optional LLM integration test executing the 4 sample questions if API key is active."""
        if not os.getenv("MISTRAL_API_KEY"):
            self.skipTest("MISTRAL_API_KEY not found in environment.")

        questions = [
            ("Who is responsible for the frontend?", "Rahul"),
            ("What database was selected?", "PostgreSQL"),
            ("When is the initial launch planned?", "October 15"),
            ("What is the deployment budget?", None),  # Expect statement saying not finalized/unknown
        ]

        print("\n--- Running VideoRAG Sample Q&A Test ---")
        for q, expected in questions:
            try:
                res = self.rag.ask_with_sources(q)
                ans = res["answer"]
                print(f"\nQ: {q}\nA: {ans}\nSources: {len(res['sources'])} chunks")
                self.assertTrue(len(ans) > 0)
                if "rate-limited" in ans.lower() or "rate limit" in ans.lower():
                    print(f"Skipping assertion for '{q}' due to Mistral rate limit.")
                    return
                if expected:
                    self.assertIn(expected.lower(), ans.lower())
                else:
                    # Should state it's not finalized or not available
                    lower_ans = ans.lower()
                    self.assertTrue(
                        "not" in lower_ans
                        or "couldn't find" in lower_ans
                        or "hasn't been" in lower_ans
                        or "yet" in lower_ans
                        or "unfinalized" in lower_ans
                    )
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    print(f"Skipping LLM assert due to Mistral rate limit: {e}")
                    return
                raise e

if __name__ == "__main__":
    unittest.main()
