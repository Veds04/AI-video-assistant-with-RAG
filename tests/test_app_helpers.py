import unittest
from utils.exporters import (
    export_transcript_txt,
    export_summary_txt,
    export_full_report_pdf,
)

class TestAppHelpers(unittest.TestCase):
    def test_export_txt(self):
        text = "This is a sample transcript."
        txt_bytes = export_transcript_txt(text)
        self.assertIsInstance(txt_bytes, bytes)
        self.assertEqual(txt_bytes.decode("utf-8"), text)

    def test_export_summary_txt(self):
        summary = "# Overview\nKey points here."
        summary_bytes = export_summary_txt(summary)
        self.assertIsInstance(summary_bytes, bytes)
        self.assertIn(b"# Overview", summary_bytes)

    def test_export_pdf_generation(self):
        title = "AI Video Meeting"
        summary = "# Overview\nProject launch meeting.\n\n# Key Takeaways\n- Postgres database."
        actions = [{"task": "Schema design", "owner": "Priya", "deadline": "Wednesday"}]
        decisions = [{"decision": "Use PostgreSQL", "context": "Production DB"}]
        questions = [{"question": "Budget?", "context": "Pending approval"}]

        pdf_bytes = export_full_report_pdf(
            title=title,
            summary=summary,
            action_items=actions,
            decisions=decisions,
            open_questions=questions,
            source="https://youtube.com/watch?v=sample",
        )

        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 500)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))

    def test_export_pdf_empty_sections(self):
        pdf_bytes = export_full_report_pdf(
            title="Empty Meeting",
            summary="",
            action_items=[],
            decisions=[],
            open_questions=[],
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))

if __name__ == "__main__":
    unittest.main()
