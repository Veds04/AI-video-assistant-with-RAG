import unittest
from core.extractor import ActionItem, Decision, OpenQuestion, MeetingInsights

class TestExtractor(unittest.TestCase):
    def test_pydantic_models(self):
        action = ActionItem(task="Do something", owner="John", deadline="Tomorrow")
        self.assertEqual(action.task, "Do something")
        self.assertEqual(action.owner, "John")
        
        meeting = MeetingInsights(
            action_items=[action],
            decisions=[],
            open_questions=[]
        )
        self.assertEqual(len(meeting.action_items), 1)

if __name__ == '__main__':
    unittest.main()
