import os
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

from utils.audio_preprocessors import download_video, convert_to_wav, split_audio
from core.transcriber import transcribe_chunks
from core.translate import translate_chunks
from core.summarizer import summarize_transcript
from core.extractor import extract_meeting_insights
from core.vector_store import (
    create_transcript_chunks,
    build_vector_store,
    generate_video_id,
)
from core.rag_engine import VideoRAG

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    DOWNLOADING = "downloading"
    CONVERTING = "converting"
    CHUNKING = "chunking"
    TRANSCRIBING = "transcribing"
    TRANSLATING = "translating"
    SUMMARIZING = "summarizing"
    EXTRACTING = "extracting"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


@dataclass
class VideoProcessingResult:
    source: str
    video_id: str
    title: str = "Video Transcript"
    audio_path: Optional[str] = None
    chunk_paths: List[str] = field(default_factory=list)
    transcript: str = ""
    translated_transcript: str = ""
    summary: str = ""
    action_items: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    open_questions: List[Dict[str, Any]] = field(default_factory=list)
    vector_store: Optional[Any] = None
    rag_engine: Optional[VideoRAG] = None
    session_dir: str = ""
    stage_errors: Dict[str, str] = field(default_factory=dict)


def assemble_transcript(texts: List[str]) -> str:
    """Combine chunk texts into an ordered, clean transcript string."""
    cleaned = []
    for i, text in enumerate(texts, 1):
        stripped = text.strip() if text else ""
        if stripped:
            cleaned.append(f"[Chunk {i}]\n{stripped}")
    return "\n\n".join(cleaned)


class VideoProcessingPipeline:
    """Coordinates video download, audio conversion, transcription, translation,
    summarization, insights extraction, and RAG index creation."""

    def __init__(
        self,
        sessions_base_dir: str = "data/sessions",
        chroma_base_dir: str = "data/chroma",
        chunk_minutes: int = 10,
    ):
        self.sessions_base_dir = os.path.abspath(sessions_base_dir)
        self.chroma_base_dir = os.path.abspath(chroma_base_dir)
        self.chunk_minutes = chunk_minutes

    def _notify(
        self,
        callback: Optional[Callable[[PipelineStage, str], None]],
        stage: PipelineStage,
        message: str,
    ) -> None:
        logger.info("[%s] %s", stage.value.upper(), message)
        if callback:
            try:
                callback(stage, message)
            except Exception as e:
                logger.warning("Progress callback raised an exception: %s", e)

    def _is_url(self, source: str) -> bool:
        src = source.strip().lower()
        return src.startswith("http://") or src.startswith("https://")

    def _derive_title(self, source: str, session_dir: str) -> str:
        """Derive a friendly title from source URL or filename."""
        if self._is_url(source):
            # Try to read downloaded video file name in session if available
            base = source.rstrip("/").split("/")[-1].split("?")[0]
            return f"Video ({base})" if base else "YouTube Video"
        base = os.path.splitext(os.path.basename(source))[0]
        return base.replace("_", " ").replace("-", " ").title()

    def process(
        self,
        source: str,
        progress_callback: Optional[Callable[[PipelineStage, str], None]] = None,
        reuse_intermediate: bool = True,
    ) -> VideoProcessingResult:
        """Execute the end-to-end processing pipeline for a given source URL or local file."""
        if not source or not source.strip():
            raise ValueError("Processing source cannot be empty.")

        source = source.strip()
        video_id = generate_video_id(source)
        session_dir = os.path.join(self.sessions_base_dir, video_id)
        os.makedirs(session_dir, exist_ok=True)

        result = VideoProcessingResult(
            source=source,
            video_id=video_id,
            title=self._derive_title(source, session_dir),
            session_dir=session_dir,
        )

        # ---------------------------------------------------------
        # 1. Download / Load Media
        # ---------------------------------------------------------
        media_path = source
        if self._is_url(source):
            self._notify(progress_callback, PipelineStage.DOWNLOADING, "Downloading media...")
            try:
                media_path = download_video(source)
                if not os.path.exists(media_path):
                    raise FileNotFoundError(f"Downloaded media file not found at: {media_path}")
            except Exception as e:
                err_msg = f"Download stage failed: {e}"
                logger.error(err_msg)
                result.stage_errors["download"] = str(e)
                self._notify(progress_callback, PipelineStage.FAILED, err_msg)
                raise RuntimeError(err_msg) from e
        else:
            if not os.path.exists(media_path):
                raise FileNotFoundError(f"Local file does not exist: {media_path}")

        # ---------------------------------------------------------
        # 2. Convert Audio to 16kHz Mono WAV
        # ---------------------------------------------------------
        self._notify(progress_callback, PipelineStage.CONVERTING, "Converting audio to 16kHz mono WAV...")
        try:
            wav_path = convert_to_wav(media_path)
            result.audio_path = wav_path
        except Exception as e:
            err_msg = f"Audio conversion stage failed: {e}"
            logger.error(err_msg)
            result.stage_errors["conversion"] = str(e)
            self._notify(progress_callback, PipelineStage.FAILED, err_msg)
            raise RuntimeError(err_msg) from e

        # ---------------------------------------------------------
        # 3. Audio Chunking
        # ---------------------------------------------------------
        self._notify(progress_callback, PipelineStage.CHUNKING, f"Splitting audio into {self.chunk_minutes}-min chunks...")
        try:
            chunks = split_audio(wav_path, chunk_minutes=self.chunk_minutes)
            if not chunks:
                raise RuntimeError("No audio chunks generated from WAV file.")
            result.chunk_paths = chunks
        except Exception as e:
            err_msg = f"Audio chunking stage failed: {e}"
            logger.error(err_msg)
            result.stage_errors["chunking"] = str(e)
            self._notify(progress_callback, PipelineStage.FAILED, err_msg)
            raise RuntimeError(err_msg) from e

        # ---------------------------------------------------------
        # 4. Whisper Transcription
        # ---------------------------------------------------------
        transcript_file = os.path.join(session_dir, "transcript.txt")
        raw_texts: List[str] = []

        if reuse_intermediate and os.path.exists(transcript_file):
            logger.info("Reusing cached transcript from: %s", transcript_file)
            with open(transcript_file, "r", encoding="utf-8") as f:
                result.transcript = f.read()
        else:
            self._notify(
                progress_callback,
                PipelineStage.TRANSCRIBING,
                f"Transcribing {len(result.chunk_paths)} audio chunk(s) with Whisper...",
            )
            try:
                raw_texts = transcribe_chunks(result.chunk_paths)
                result.transcript = assemble_transcript(raw_texts)
                with open(transcript_file, "w", encoding="utf-8") as f:
                    f.write(result.transcript)
            except Exception as e:
                err_msg = f"Transcription stage failed: {e}"
                logger.error(err_msg)
                result.stage_errors["transcription"] = str(e)
                self._notify(progress_callback, PipelineStage.FAILED, err_msg)
                raise RuntimeError(err_msg) from e

        if not result.transcript.strip():
            logger.warning("Transcript is completely empty. Halting downstream LLM processing.")
            self._notify(progress_callback, PipelineStage.READY, "Completed with empty transcript.")
            return result

        # ---------------------------------------------------------
        # 5. Sarvam AI Translation
        # ---------------------------------------------------------
        translated_file = os.path.join(session_dir, "translated_transcript.txt")
        if reuse_intermediate and os.path.exists(translated_file):
            logger.info("Reusing cached translated transcript from: %s", translated_file)
            with open(translated_file, "r", encoding="utf-8") as f:
                result.translated_transcript = f.read()
        else:
            self._notify(progress_callback, PipelineStage.TRANSLATING, "Translating transcript to English with Sarvam AI...")
            try:
                # If raw_texts is empty because we didn't transcribe just now, reconstruct from chunks or full transcript
                if not raw_texts:
                    raw_texts = [result.transcript]
                translated_chunks = translate_chunks(raw_texts)
                result.translated_transcript = assemble_transcript(translated_chunks)
                if not result.translated_transcript.strip():
                    # Fallback to original transcript if translation yielded empty output
                    result.translated_transcript = result.transcript
                with open(translated_file, "w", encoding="utf-8") as f:
                    f.write(result.translated_transcript)
            except Exception as e:
                logger.warning("Translation stage encountered an error: %s. Falling back to original transcript.", e)
                result.stage_errors["translation"] = str(e)
                result.translated_transcript = result.transcript

        final_english_transcript = (
            result.translated_transcript if result.translated_transcript.strip() else result.transcript
        )

        # ---------------------------------------------------------
        # 6. Summarization (Map-Reduce with Mistral)
        # ---------------------------------------------------------
        summary_file = os.path.join(session_dir, "summary.txt")
        if reuse_intermediate and os.path.exists(summary_file):
            with open(summary_file, "r", encoding="utf-8") as f:
                result.summary = f.read()
        else:
            self._notify(progress_callback, PipelineStage.SUMMARIZING, "Generating structured summary with Mistral...")
            try:
                result.summary = summarize_transcript(final_english_transcript)
                with open(summary_file, "w", encoding="utf-8") as f:
                    f.write(result.summary)
            except Exception as e:
                logger.error("Summarization stage failed: %s", e)
                result.stage_errors["summarization"] = str(e)
                result.summary = "Summary generation failed or rate limit reached."

        # ---------------------------------------------------------
        # 7. Meeting Insights Extraction
        # ---------------------------------------------------------
        insights_file = os.path.join(session_dir, "insights.json")
        if reuse_intermediate and os.path.exists(insights_file):
            try:
                with open(insights_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    result.action_items = data.get("action_items", [])
                    result.decisions = data.get("decisions", [])
                    result.open_questions = data.get("open_questions", [])
            except Exception as e:
                logger.warning("Failed to load cached insights: %s", e)

        if not result.action_items and not result.decisions and not result.open_questions:
            self._notify(progress_callback, PipelineStage.EXTRACTING, "Extracting action items, decisions, and questions...")
            try:
                insights = extract_meeting_insights(final_english_transcript)
                result.action_items = insights.get("action_items", [])
                result.decisions = insights.get("decisions", [])
                result.open_questions = insights.get("open_questions", [])
                with open(insights_file, "w", encoding="utf-8") as f:
                    json.dump(insights, f, indent=2)
            except Exception as e:
                logger.error("Insights extraction failed: %s", e)
                result.stage_errors["extraction"] = str(e)

        # ---------------------------------------------------------
        # 8. Vector Store & RAG Initialization
        # ---------------------------------------------------------
        self._notify(progress_callback, PipelineStage.INDEXING, "Building Chroma vector store...")
        try:
            chunks_docs = create_transcript_chunks(
                final_english_transcript,
                source=result.title,
            )
            if chunks_docs:
                vstore = build_vector_store(
                    documents=chunks_docs,
                    video_id=video_id,
                    persist_directory=self.chroma_base_dir,
                )
                result.vector_store = vstore
                result.rag_engine = VideoRAG(vector_store=vstore, k=4)
        except Exception as e:
            logger.error("Vector store indexing failed: %s", e)
            result.stage_errors["indexing"] = str(e)

        # ---------------------------------------------------------
        # 9. Ready
        # ---------------------------------------------------------
        self._notify(progress_callback, PipelineStage.READY, "Processing complete! Assistant is ready.")
        return result
