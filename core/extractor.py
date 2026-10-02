import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
logger = logging.getLogger(__name__)


class ActionItem(BaseModel):
    task: str = Field(description="The task to be completed")
    owner: str = Field(description="The person responsible. If not specified, return 'Not specified'")
    deadline: str = Field(description="The deadline for the task. If not specified, return 'Not specified'")


class Decision(BaseModel):
    decision: str = Field(description="The decision that was made")
    context: str = Field(description="Context or reasoning behind the decision")


class OpenQuestion(BaseModel):
    question: str = Field(description="The open question raised")
    context: str = Field(description="Context surrounding the question")


class MeetingInsights(BaseModel):
    action_items: List[ActionItem] = Field(default_factory=list, description="List of action items")
    decisions: List[Decision] = Field(default_factory=list, description="List of decisions made")
    open_questions: List[OpenQuestion] = Field(default_factory=list, description="List of open questions")


def get_llm() -> ChatMistralAI:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment variables.")
    model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    return ChatMistralAI(api_key=api_key.strip(), model=model.strip(), temperature=0.0)


SYSTEM_PROMPT = """You extract structured meeting information from transcripts.
Only extract information explicitly stated or clearly established in the transcript.
Never invent owners, deadlines, decisions, or questions.
If an owner or deadline is not specified, return 'Not specified'.
Return valid structured data."""


def _deduplicate_items(items: List[Dict[str, Any]], key_fn) -> List[Dict[str, Any]]:
    """Deduplicate dict items preserving order."""
    seen = set()
    deduped = []
    for item in items:
        k = key_fn(item)
        if k not in seen:
            seen.add(k)
            deduped.append(item)
    return deduped


def extract_meeting_insights(transcript: str) -> dict:
    """Extract action items, decisions, and open questions from a transcript with deduplication."""
    if not transcript or not transcript.strip():
        return {"action_items": [], "decisions": [], "open_questions": []}

    logger.info("Extracting meeting insights...")

    from core.summarizer import split_transcript
    chunks = split_transcript(transcript, max_chars=8000)

    llm = get_llm()
    structured_llm = llm.with_structured_output(MeetingInsights)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Extract insights from this text:\n\n{text}"),
    ])

    chain = prompt | structured_llm

    all_action_items: List[Dict[str, Any]] = []
    all_decisions: List[Dict[str, Any]] = []
    all_open_questions: List[Dict[str, Any]] = []

    for i, chunk in enumerate(chunks, 1):
        logger.info("Extracting insights from chunk %d/%d...", i, len(chunks))
        try:
            result = chain.invoke({"text": chunk})
            if result:
                for item in result.action_items:
                    all_action_items.append(item.model_dump())
                for item in result.decisions:
                    all_decisions.append(item.model_dump())
                for item in result.open_questions:
                    all_open_questions.append(item.model_dump())
        except Exception as e:
            logger.error("Extraction failed for chunk %d: %s", i, e)
            raise RuntimeError(f"Extraction failed for chunk {i}: {e}") from e

    # Deduplicate across chunk boundaries
    deduped_actions = _deduplicate_items(
        all_action_items,
        lambda x: (x.get("task", "").strip().lower(), x.get("owner", "").strip().lower()),
    )
    deduped_decisions = _deduplicate_items(
        all_decisions,
        lambda x: x.get("decision", "").strip().lower(),
    )
    deduped_questions = _deduplicate_items(
        all_open_questions,
        lambda x: x.get("question", "").strip().lower(),
    )

    return {
        "action_items": deduped_actions,
        "decisions": deduped_decisions,
        "open_questions": deduped_questions,
    }


def extract_action_items(transcript: str) -> list[dict]:
    insights = extract_meeting_insights(transcript)
    return insights.get("action_items", [])


def extract_decisions(transcript: str) -> list[dict]:
    insights = extract_meeting_insights(transcript)
    return insights.get("decisions", [])


def extract_questions(transcript: str) -> list[dict]:
    insights = extract_meeting_insights(transcript)
    return insights.get("open_questions", [])
