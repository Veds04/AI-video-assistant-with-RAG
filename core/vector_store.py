import os
import re
import hashlib
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

# Module-level cache for embedding model to avoid reloading
_EMBEDDINGS: Optional[HuggingFaceEmbeddings] = None
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_function() -> HuggingFaceEmbeddings:
    """Load or return cached HuggingFaceEmbeddings model."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        logger.info("Initializing embedding model: %s", EMBEDDING_MODEL_NAME)
        try:
            _EMBEDDINGS = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Embedding model initialized successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HuggingFace embeddings: {e}")
    return _EMBEDDINGS


def generate_video_id(source_identifier: str) -> str:
    """Generate a clean, deterministic video/session identifier."""
    if not source_identifier or not source_identifier.strip():
        source_identifier = "default_session"
    clean_hash = hashlib.sha256(source_identifier.strip().encode("utf-8")).hexdigest()[:12]
    # Keep safe alphanumeric name
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", os.path.basename(source_identifier))[:20]
    return f"{safe_name}_{clean_hash}".strip("_")


def create_transcript_chunks(
    transcript: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    source: str = "transcript",
) -> List[Document]:
    """Split transcript text into meaningful Document chunks with metadata.

    Preserves chunk indices and any chunk/timestamp headers present in text.
    """
    if not transcript or not transcript.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )

    raw_chunks = splitter.split_text(transcript.strip())
    documents: List[Document] = []

    for idx, text in enumerate(raw_chunks, start=1):
        # Look for [Chunk X] or timestamp patterns in text if available
        meta = {
            "chunk_id": idx,
            "source": source,
            "char_count": len(text),
        }
        chunk_match = re.search(r"\[Chunk\s+(\d+)\]", text)
        if chunk_match:
            meta["original_chunk"] = int(chunk_match.group(1))

        # Check for timestamp patterns like [00:15] or (01:23)
        time_match = re.findall(r"\[?\b(\d{1,2}:\d{2}(?::\d{2})?)\b\]?", text)
        if time_match:
            meta["timestamps"] = time_match

        documents.append(Document(page_content=text, metadata=meta))

    logger.info("Created %d transcript chunks (size=%d, overlap=%d).", len(documents), chunk_size, chunk_overlap)
    return documents


def sanitize_collection_name(video_id: str) -> str:
    """Sanitize collection name to strictly satisfy Chroma naming constraints (3-63 chars, alphanumeric start/end)."""
    clean_id = re.sub(r"[^a-zA-Z0-9_-]", "_", video_id)
    collection_name = f"vid_{clean_id}"[:63].strip("_-")
    if len(collection_name) < 3:
        collection_name = f"vid_{clean_id}".ljust(3, "0")
    if not collection_name[0].isalnum():
        collection_name = "v" + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + "0"
    return collection_name[:63]


def build_vector_store(
    documents: List[Document],
    video_id: str = "default_video",
    persist_directory: str = "data/chroma",
) -> Chroma:
    """Build and persist an isolated Chroma collection for a specific video/session."""
    if not documents:
        raise ValueError("Cannot build vector store with empty document list.")

    collection_name = sanitize_collection_name(video_id)
    clean_id = re.sub(r"[^a-zA-Z0-9_-]", "_", video_id).strip("_") or "session"
    video_persist_dir = os.path.abspath(os.path.join(persist_directory, clean_id))
    os.makedirs(video_persist_dir, exist_ok=True)

    embeddings = get_embedding_function()

    try:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=video_persist_dir,
        )
        logger.info(
            "Vector store successfully built in '%s' with collection '%s'.",
            video_persist_dir,
            collection_name,
        )
        return vector_store
    except Exception as e:
        raise RuntimeError(f"ChromaDB initialization failed for video '{video_id}': {e}")


def load_vector_store(
    video_id: str,
    persist_directory: str = "data/chroma",
) -> Chroma:
    """Load an existing Chroma collection for a specific video_id."""
    collection_name = sanitize_collection_name(video_id)
    clean_id = re.sub(r"[^a-zA-Z0-9_-]", "_", video_id).strip("_") or "session"
    video_persist_dir = os.path.abspath(os.path.join(persist_directory, clean_id))
    if not os.path.exists(video_persist_dir):
        raise FileNotFoundError(f"No persisted vector store found for video '{video_id}' at '{video_persist_dir}'")

    embeddings = get_embedding_function()
    try:
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=video_persist_dir,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load vector store for video '{video_id}': {e}")


def get_retriever(vector_store: Chroma, k: int = 4):
    """Return a retriever configured for similarity search with k chunks."""
    if vector_store is None:
        raise ValueError("Cannot create retriever from None vector store.")
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
