import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from core.pipeline import (
    VideoProcessingPipeline,
    VideoProcessingResult,
    PipelineStage,
    assemble_transcript,
)

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sessions_dir = os.path.join(self.test_dir, "sessions")
        self.chroma_dir = os.path.join(self.test_dir, "chroma")
        self.pipeline = VideoProcessingPipeline(
            sessions_base_dir=self.sessions_dir,
            chroma_base_dir=self.chroma_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_assemble_transcript(self):
        chunks = ["Hello world.", "This is chunk 2.", ""]
        assembled = assemble_transcript(chunks)
        self.assertIn("[Chunk 1]\nHello world.", assembled)
        self.assertIn("[Chunk 2]\nThis is chunk 2.", assembled)
        self.assertNotIn("[Chunk 3]", assembled)

    def test_empty_source_validation(self):
        with self.assertRaises(ValueError):
            self.pipeline.process("")

    def test_missing_local_file_failure(self):
        with self.assertRaises(FileNotFoundError):
            self.pipeline.process("non_existent_video_file.mp4")

    @patch("core.pipeline.convert_to_wav")
    @patch("core.pipeline.split_audio")
    @patch("core.pipeline.transcribe_chunks")
    @patch("core.pipeline.translate_chunks")
    @patch("core.pipeline.summarize_transcript")
    @patch("core.pipeline.extract_meeting_insights")
    @patch("core.pipeline.build_vector_store")
    def test_pipeline_mocked_end_to_end(
        self,
        mock_build_vs,
        mock_extract,
        mock_summarize,
        mock_translate,
        mock_transcribe,
        mock_split,
        mock_convert,
    ):
        # Setup temporary dummy input file
        dummy_media = os.path.join(self.test_dir, "dummy.mp4")
        with open(dummy_media, "w") as f:
            f.write("mock media")

        mock_convert.return_value = os.path.join(self.test_dir, "dummy.wav")
        mock_split.return_value = [os.path.join(self.test_dir, "chunk_001.wav")]
        mock_transcribe.return_value = ["Hindi or Hinglish discussion about launch."]
        mock_translate.return_value = ["Discussion about launch in English."]
        mock_summarize.return_value = "# Overview\nLaunch summary."
        mock_extract.return_value = {
            "action_items": [{"task": "Deploy", "owner": "Alice", "deadline": "Friday"}],
            "decisions": [{"decision": "Use PostgreSQL", "context": "Production DB"}],
            "open_questions": [{"question": "Budget?", "context": "Pending approval"}],
        }
        mock_vs_instance = MagicMock()
        mock_build_vs.return_value = mock_vs_instance

        stages_recorded = []
        def record_stage(stg, msg):
            stages_recorded.append(stg)

        result = self.pipeline.process(
            source=dummy_media,
            progress_callback=record_stage,
            reuse_intermediate=False,
        )

        # Verify returned result
        self.assertIsInstance(result, VideoProcessingResult)
        self.assertEqual(result.source, dummy_media)
        self.assertIn("Discussion about launch in English.", result.translated_transcript)
        self.assertIn("# Overview", result.summary)
        self.assertEqual(len(result.action_items), 1)
        self.assertEqual(result.action_items[0]["owner"], "Alice")
        self.assertIsNotNone(result.rag_engine)

        # Verify pipeline stages executed in correct order
        expected_sequence = [
            PipelineStage.CONVERTING,
            PipelineStage.CHUNKING,
            PipelineStage.TRANSCRIBING,
            PipelineStage.TRANSLATING,
            PipelineStage.SUMMARIZING,
            PipelineStage.EXTRACTING,
            PipelineStage.INDEXING,
            PipelineStage.READY,
        ]
        self.assertEqual(stages_recorded, expected_sequence)

        # Verify session files were persisted on disk
        self.assertTrue(os.path.exists(os.path.join(result.session_dir, "transcript.txt")))
        self.assertTrue(os.path.exists(os.path.join(result.session_dir, "translated_transcript.txt")))
        self.assertTrue(os.path.exists(os.path.join(result.session_dir, "summary.txt")))
        self.assertTrue(os.path.exists(os.path.join(result.session_dir, "insights.json")))

if __name__ == "__main__":
    unittest.main()
