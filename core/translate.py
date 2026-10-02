import os
import re
import logging
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_CLIENT: Optional[object] = None


def get_sarvam_client():
    """Lazily initialize SarvamAI client."""
    global _CLIENT
    if _CLIENT is None:
        api_key = os.getenv("SARVAM_API_KEY")
        if not api_key:
            logger.warning("SARVAM_API_KEY is not set. Translation will be skipped.")
            return None
        try:
            from sarvamai import SarvamAI
            _CLIENT = SarvamAI(api_subscription_key=api_key.strip())
        except Exception as e:
            logger.error("Failed to initialize SarvamAI client: %s", e)
            return None
    return _CLIENT


def chunk_text_safely(text: str, max_chars: int = 900) -> List[str]:
    """Split text into chunks without cutting words or sentences in half."""
    if not text or not text.strip():
        return []

    # Split into sentences or lines first
    sentences = re.split(r"(?<=[.!?\n])\s+", text.strip())
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If a single sentence exceeds max_chars, split by words
        if len(sentence) > max_chars:
            words = sentence.split()
            for word in words:
                if current_len + len(word) + 1 > max_chars:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_len = 0
                current_chunk.append(word)
                current_len += len(word) + 1
        elif current_len + len(sentence) + 1 > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
            current_chunk.append(sentence)
            current_len += len(sentence) + 1
        else:
            current_chunk.append(sentence)
            current_len += len(sentence) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def translate_text(text: str) -> str:
    """Translate Hindi/English/mixed text into clear English safely."""
    if not text or not text.strip():
        return ""

    client = get_sarvam_client()
    if client is None:
        # Fall back to original text if Sarvam client is unavailable
        return text.strip()

    try:
        parts = chunk_text_safely(text, max_chars=900)
        translated: List[str] = []

        for part in parts:
            if not part.strip():
                continue
            response = client.text.translate(
                input=part,
                source_language_code="auto",
                target_language_code="en-IN",
                model="mayura:v1",
                mode="formal",
            )
            trans_result = getattr(response, "translated_text", "")
            if trans_result:
                translated.append(trans_result)
            else:
                translated.append(part)

        return " ".join(translated)

    except Exception as e:
        logger.warning("Translation failed: %s. Using original text.", e)
        # Graceful fallback: return original text rather than crashing pipeline
        return text.strip()


def translate_chunks(texts: List[str]) -> List[str]:
    """Translate all transcript chunks into English."""
    results: List[str] = []
    for i, text in enumerate(texts, 1):
        logger.info("Translating chunk %d/%d...", i, len(texts))
        results.append(translate_text(text))
    return results