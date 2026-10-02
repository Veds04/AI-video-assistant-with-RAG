import os
import shutil
import tempfile
import unittest
from core.vector_store import (
    create_transcript_chunks,
    build_vector_store,
    get_retriever,
    generate_video_id,
)

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_empty_transcript(self):
        chunks = create_transcript_chunks("")
        self.assertEqual(chunks, [])
        chunks_spaces = create_transcript_chunks("   \n\t  ")
        self.assertEqual(chunks_spaces, [])

    def test_chunk_generation_and_metadata(self):
        sample = "The quick brown fox jumps over the lazy dog. " * 30
        chunks = create_transcript_chunks(sample, chunk_size=200, chunk_overlap=30, source="my_test")
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["chunk_id"], 1)
        self.assertEqual(chunks[0].metadata["source"], "my_test")
        self.assertIn("char_count", chunks[0].metadata)

    def test_vector_store_creation_and_retrieval(self):
        transcript = """
        Rahul will finish the frontend by Friday.
        Priya will prepare the database schema by Wednesday.
        The team decided to use PostgreSQL for the production database.
        The initial launch is planned for October 15.
        """
        chunks = create_transcript_chunks(transcript)
        self.assertGreater(len(chunks), 0)

        vid_id = generate_video_id("test_video_1")
        vstore = build_vector_store(chunks, video_id=vid_id, persist_directory=self.test_dir)
        retriever = get_retriever(vstore, k=2)

        results = retriever.invoke("What database was chosen?")
        self.assertGreater(len(results), 0)
        found_postgres = any("PostgreSQL" in doc.page_content for doc in results)
        self.assertTrue(found_postgres)

    def test_video_isolation(self):
        # Video 1: Project Apollo talks about Mars
        v1_text = "Mission Apollo is focused on launching a rover to Mars in November."
        v1_chunks = create_transcript_chunks(v1_text)
        v1_store = build_vector_store(v1_chunks, video_id="video_apollo", persist_directory=self.test_dir)

        # Video 2: Project Artemis talks about the Moon
        v2_text = "Mission Artemis is focused on landing astronauts on the Moon next year."
        v2_chunks = create_transcript_chunks(v2_text)
        v2_store = build_vector_store(v2_chunks, video_id="video_artemis", persist_directory=self.test_dir)

        # Query Apollo store
        apollo_retriever = get_retriever(v1_store, k=2)
        apollo_results = apollo_retriever.invoke("Where are we landing?")
        apollo_text = " ".join([d.page_content for d in apollo_results])
        self.assertIn("Mars", apollo_text)
        self.assertNotIn("Moon", apollo_text)

        # Query Artemis store
        artemis_retriever = get_retriever(v2_store, k=2)
        artemis_results = artemis_retriever.invoke("Where are we landing?")
        artemis_text = " ".join([d.page_content for d in artemis_results])
        self.assertIn("Moon", artemis_text)
        self.assertNotIn("Mars", artemis_text)

    def test_sanitize_collection_name(self):
        from core.vector_store import sanitize_collection_name
        name1 = sanitize_collection_name("video_ending_with_underscore_")
        self.assertTrue(name1[0].isalnum())
        self.assertTrue(name1[-1].isalnum())
        self.assertNotIn("__", name1)

        name2 = sanitize_collection_name("a" * 100)
        self.assertLessEqual(len(name2), 63)
        self.assertTrue(name2[-1].isalnum())

        name3 = sanitize_collection_name("!@#")
        self.assertGreaterEqual(len(name3), 3)
        self.assertTrue(name3[0].isalnum())
        self.assertTrue(name3[-1].isalnum())

if __name__ == "__main__":
    unittest.main()
