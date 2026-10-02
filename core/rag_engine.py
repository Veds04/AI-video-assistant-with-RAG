import os
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma

from core.vector_store import (
    create_transcript_chunks,
    build_vector_store,
    get_retriever,
    generate_video_id,
)

load_dotenv()
logger = logging.getLogger(__name__)

RAG_PROMPT_TEMPLATE = """You are an assistant answering questions about a video transcript.

Answer ONLY using the provided transcript context.

Rules:
- Do not use outside knowledge.
- Do not invent information.
- Do not make assumptions that aren't supported by the context.
- If the answer is not contained in the context, say:
  'I couldn't find that information in the transcript.'
- Keep the answer concise and directly answer the question.

Context:
{context}

Question:
{question}"""


def get_llm() -> ChatMistralAI:
    """Instantiate and return ChatMistralAI."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment variables.")
    model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    return ChatMistralAI(api_key=api_key, model=model, temperature=0.0)


def format_docs(docs: List[Document]) -> str:
    """Format a list of Document objects into a clean text block."""
    if not docs:
        return "No relevant context found."
    formatted = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id", "?")
        timestamps = doc.metadata.get("timestamps")
        header = f"[Chunk {chunk_id}]"
        if timestamps:
            header += f" [Time: {', '.join(timestamps)}]"
        formatted.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n".join(formatted)


class VideoRAG:
    """Encapsulates Chroma vector store, retriever, LLM, and LCEL RAG chain for a video."""

    def __init__(self, vector_store: Chroma, k: int = 4):
        if vector_store is None:
            raise ValueError("vector_store cannot be None.")
        self.vector_store = vector_store
        self.k = k
        self.retriever = get_retriever(vector_store, k=k)
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    @classmethod
    def from_transcript(
        cls,
        transcript: str,
        video_id: Optional[str] = None,
        source: str = "transcript",
        k: int = 4,
        persist_directory: str = "data/chroma",
    ) -> "VideoRAG":
        """Convenience factory to create a VideoRAG directly from raw transcript text."""
        if not transcript or not transcript.strip():
            raise ValueError("Cannot initialize VideoRAG with an empty transcript.")

        vid = video_id or generate_video_id(transcript[:100])
        documents = create_transcript_chunks(transcript, source=source)
        vector_store = build_vector_store(
            documents=documents,
            video_id=vid,
            persist_directory=persist_directory,
        )
        return cls(vector_store=vector_store, k=k)

    def retrieve_context(self, question: str) -> List[Document]:
        """Retrieve relevant context chunks for a question."""
        if not question or not question.strip():
            return []
        try:
            return self.retriever.invoke(question.strip())
        except Exception as e:
            logger.error("Context retrieval failed: %s", e)
            raise RuntimeError(f"Context retrieval failed: {e}")

    def ask(self, question: str) -> str:
        """Query the video transcript and return the grounded answer."""
        result = self.ask_with_sources(question)
        return result["answer"]

    def ask_with_sources(self, question: str) -> Dict[str, Any]:
        """Query the video transcript, returning grounded answer and source documents."""
        if not question or not question.strip():
            return {
                "answer": "Please provide a valid question.",
                "sources": [],
            }

        docs = self.retrieve_context(question)
        if not docs:
            return {
                "answer": "I couldn't find that information in the transcript.",
                "sources": [],
            }

        formatted_context = format_docs(docs)

        sources_meta = [
            {
                "chunk_id": doc.metadata.get("chunk_id"),
                "source": doc.metadata.get("source", "transcript"),
                "timestamps": doc.metadata.get("timestamps", []),
                "content": doc.page_content,
            }
            for doc in docs
        ]

        try:
            llm = get_llm()
            # LCEL chain: prompt | llm | StrOutputParser()
            chain = self.prompt | llm | StrOutputParser()
            answer = chain.invoke({
                "context": formatted_context,
                "question": question.strip(),
            })

            return {
                "answer": answer.strip(),
                "sources": sources_meta,
            }
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate limit" in err_str.lower():
                logger.warning("Mistral rate limit hit during RAG query: %s", e)
                return {
                    "answer": "The AI assistant is temporarily rate-limited. Please wait a moment and try asking again.",
                    "sources": sources_meta,
                }
            logger.error("RAG query failed: %s", e)
            raise RuntimeError(f"RAG query failed: {e}") from e
