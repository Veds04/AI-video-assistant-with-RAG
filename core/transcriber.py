import os
import logging
from typing import List, Optional
import whisper

logger = logging.getLogger(__name__)

_model: Optional[object] = None
_loaded_model_name: Optional[str] = None


def load_model():
    """Load Whisper once and reuse the model, or reload if model name changed."""
    global _model, _loaded_model_name

    requested_model = os.getenv("WHISPER_MODEL", "small").strip()

    if _model is None or _loaded_model_name != requested_model:
        logger.info("Loading Whisper '%s' model...", requested_model)
        _model = whisper.load_model(requested_model)
        _loaded_model_name = requested_model
        logger.info("Whisper model loaded successfully.")

    return _model


def transcribe_chunk(audio_path: str) -> str:
    """Transcribe or translate an audio chunk into English."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio chunk not found: {audio_path}")

    model = load_model()

    try:
        result = model.transcribe(
            audio_path,
            task="translate",
            fp16=False,
        )
        return result.get("text", "").strip()
    except Exception as e:
        logger.error("Whisper transcription failed on %s: %s", audio_path, e)
        raise RuntimeError(f"Whisper transcription failed: {e}") from e


def transcribe_chunks(chunk_paths: List[str]) -> List[str]:
    """Process all audio chunks and return English texts."""
    if not chunk_paths:
        return []

    texts: List[str] = []
    for i, path in enumerate(chunk_paths, 1):
        logger.info("Transcribing chunk %d/%d (%s)...", i, len(chunk_paths), os.path.basename(path))
        text = transcribe_chunk(path)
        texts.append(text)

    return texts