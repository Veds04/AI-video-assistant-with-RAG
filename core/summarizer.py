import os
import logging
from typing import List
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a meeting/video transcript summarization assistant.

Summarize only information explicitly present in the transcript.
Do not hallucinate.
Do not add outside knowledge.
Do not treat assumptions as facts.

Preserve important:
- people
- dates
- numbers
- technical terms
- decisions
- commitments

Produce concise, useful output."""


def get_llm() -> ChatMistralAI:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment variables.")
    model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    return ChatMistralAI(api_key=api_key.strip(), model=model.strip(), temperature=0.0)


def split_transcript(text: str, max_chars: int = 6000) -> List[str]:
    """Split transcript into chunks without breaking sentences."""
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=500,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    return splitter.split_text(text.strip())


def summarize_chunk(text: str) -> str:
    """Produce a concise summary of one transcript chunk."""
    if not text or not text.strip():
        return ""

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Summarize this transcript chunk:\n\n{text}"),
    ])

    chain = prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"text": text.strip()})
    except Exception as e:
        logger.error("Chunk summarization failed: %s", e)
        raise RuntimeError(f"Chunk summarization failed: {e}") from e


def summarize_transcript(transcript: str) -> str:
    """Summarize transcript using map-reduce style, or direct structured summary for single chunk."""
    if not transcript or not transcript.strip():
        return ""

    logger.info("Summarizing transcript...")
    chunks = split_transcript(transcript)
    if not chunks:
        return ""

    llm = get_llm()
    structured_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", (
            "Here is the transcript content:\n\n{text}\n\n"
            "Provide a comprehensive structured summary. Your output MUST be structured EXACTLY with the following headings:\n\n"
            "# Overview\n\n"
            "# Main Topics\n\n"
            "# Important Points\n\n"
            "# Key Takeaways"
        )),
    ])
    chain = structured_prompt | llm | StrOutputParser()

    # Optimization: If the transcript fits in a single chunk, generate final structured summary in 1 LLM call!
    if len(chunks) == 1:
        logger.info("Single chunk detected: Generating structured summary in one optimized LLM call...")
        try:
            summary = chain.invoke({"text": chunks[0]})
            logger.info("Summary generated successfully.")
            return summary
        except Exception as e:
            logger.error("Direct summarization failed: %s", e)
            raise RuntimeError(f"Summarization failed: {e}") from e

    # Multi-chunk map-reduce
    logger.info("Multiple chunks detected (%d chunks): Running map-reduce summarization...", len(chunks))
    partial_summaries = []
    for i, chunk in enumerate(chunks, 1):
        logger.info("Processing chunk %d/%d...", i, len(chunks))
        part = summarize_chunk(chunk)
        partial_summaries.append(part)

    combined = "\n\n".join([f"Part {i}:\n{s}" for i, s in enumerate(partial_summaries, 1)])

    logger.info("Generating final structured summary from combined partial summaries...")
    try:
        final_summary = chain.invoke({"text": combined})
        logger.info("Summary generated successfully.")
        return final_summary
    except Exception as e:
        logger.error("Final summarization failed: %s", e)
        raise RuntimeError(f"Final summarization failed: {e}") from e
