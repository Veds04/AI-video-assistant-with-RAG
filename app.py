import os
import re
import shutil
import tempfile
import streamlit as st
from dotenv import load_dotenv

MAX_UPLOAD_SIZE_MB = 500

from core.pipeline import VideoProcessingPipeline, PipelineStage, VideoProcessingResult
from utils.exporters import export_transcript_txt, export_summary_txt, export_full_report_pdf

load_dotenv()

st.set_page_config(
    page_title="AI Video Assistant with RAG",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing_error" not in st.session_state:
    st.session_state.processing_error = None


def render_sidebar():
    with st.sidebar:
        st.title("⚙️ System Config")
        st.markdown("---")
        st.markdown("**Core Models**")
        st.caption(f"• **Whisper:** `{os.getenv('WHISPER_MODEL', 'small')}` (Local)")
        st.caption("• **Embeddings:** `all-MiniLM-L6-v2` (Local)")
        st.caption(f"• **LLM:** `{os.getenv('MISTRAL_MODEL', 'mistral-small-latest')}`")
        st.caption("• **Translation:** `Sarvam AI (mayura:v1)`")
        st.caption("• **Vector DB:** `ChromaDB`")
        st.markdown("---")
        st.markdown("**Architecture Flow**")
        st.caption("1. Video/Audio Input")
        st.caption("2. 16kHz Mono WAV conversion")
        st.caption("3. 10-Minute chunking")
        st.caption("4. Whisper transcription")
        st.caption("5. Sarvam AI translation")
        st.caption("6. Map-Reduce summarization")
        st.caption("7. Insights extraction")
        st.caption("8. ChromaDB RAG Engine")
        st.markdown("---")
        if st.session_state.result:
            if st.button("🔄 Reset / Clear Session", use_container_width=True):
                st.session_state.result = None
                st.session_state.chat_history = []
                st.session_state.processing_error = None
                st.rerun()


def render_header():
    st.title("🎬 AI Video Assistant")
    st.markdown("##### *Turn long videos and meetings into searchable knowledge.*")
    st.markdown("")


def render_input_section():
    st.subheader("1. Select Video Source")
    input_type = st.radio(
        "Choose source type:",
        ["YouTube URL", "Upload File"],
        horizontal=True,
    )

    source_path = None
    ready_to_process = False

    if input_type == "YouTube URL":
        youtube_url = st.text_input(
            "Paste YouTube Video URL:",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        if youtube_url and youtube_url.strip():
            source_path = youtube_url.strip()
            ready_to_process = True
    else:
        uploaded_file = st.file_uploader(
            "Upload audio or video file:",
            type=["mp4", "mkv", "mov", "webm", "mp3", "wav", "m4a", "flac"],
            help="Supported formats: MP4, MKV, MOV, WEBM, MP3, WAV, M4A, FLAC",
        )
        if uploaded_file is not None:
            # Validate file size
            if uploaded_file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                st.error(f"File too large. Maximum upload size is {MAX_UPLOAD_SIZE_MB} MB.")
            else:
                downloads_dir = os.path.abspath("downloads")
                os.makedirs(downloads_dir, exist_ok=True)
                # Sanitize filename: strip path components and restrict to safe characters
                safe_name = os.path.basename(uploaded_file.name)
                safe_name = re.sub(r'[^\w.\-() ]', '_', safe_name)
                if not safe_name:
                    safe_name = "uploaded_file"
                saved_path = os.path.join(downloads_dir, safe_name)
                # Verify path is still inside downloads_dir after join
                if not os.path.abspath(saved_path).startswith(downloads_dir):
                    st.error("Invalid filename.")
                else:
                    # Avoid re-writing large media files on every Streamlit rerun
                    if not os.path.exists(saved_path) or os.path.getsize(saved_path) != uploaded_file.size:
                        with open(saved_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    source_path = saved_path
                    ready_to_process = True

    st.markdown("")
    col1, col2 = st.columns([1, 4])
    with col1:
        process_clicked = st.button(
            "🚀 Process Video",
            type="primary",
            use_container_width=True,
            disabled=not ready_to_process,
        )

    return process_clicked, source_path


def execute_pipeline(source_path: str):
    pipeline = VideoProcessingPipeline()
    status_container = st.status("Initializing video pipeline...", expanded=True)

    stage_labels = {
        PipelineStage.DOWNLOADING: "⬇️ Downloading media...",
        PipelineStage.CONVERTING: "🔄 Converting audio to 16kHz mono WAV...",
        PipelineStage.CHUNKING: "✂️ Splitting audio into chunks...",
        PipelineStage.TRANSCRIBING: "🎙️ Transcribing with local Whisper...",
        PipelineStage.TRANSLATING: "🌐 Translating transcript with Sarvam AI...",
        PipelineStage.SUMMARIZING: "📝 Generating structured summary...",
        PipelineStage.EXTRACTING: "💡 Extracting action items, decisions, and questions...",
        PipelineStage.INDEXING: "🧠 Building Chroma vector index...",
        PipelineStage.READY: "✅ Ready!",
        PipelineStage.FAILED: "❌ Pipeline failed.",
    }

    def progress_callback(stage: PipelineStage, message: str):
        label = stage_labels.get(stage, message)
        status_container.write(f"**[{stage.value.upper()}]** {label}")

    try:
        result = pipeline.process(
            source=source_path,
            progress_callback=progress_callback,
            reuse_intermediate=True,
        )
        status_container.update(label="Video Processing Complete!", state="complete", expanded=False)
        st.session_state.result = result
        st.session_state.chat_history = []
        st.session_state.processing_error = None
        st.rerun()
    except Exception as e:
        import logging
        logging.getLogger(__name__).error("Pipeline failed: %s", e, exc_info=True)
        status_container.update(label="Processing Failed", state="error", expanded=True)
        st.session_state.processing_error = "Video processing failed. Check logs for details."
        st.error("⚠️ Video processing encountered an error. Please check your inputs and API keys, then try again.")


def render_overview_tab(result: VideoProcessingResult):
    st.subheader("📌 Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Source", "URL" if result.source.startswith("http") else "Local File")
    with col2:
        st.metric("Action Items", len(result.action_items))
    with col3:
        st.metric("Decisions", len(result.decisions))
    with col4:
        st.metric("Open Questions", len(result.open_questions))

    st.markdown("---")
    st.markdown(f"**Video Title / Identifier:** `{result.title}`")
    st.markdown(f"**Session ID:** `{result.video_id}`")
    if result.chunk_paths:
        st.markdown(f"**Processed Audio Chunks:** `{len(result.chunk_paths)} chunk(s)`")


def render_transcript_tab(result: VideoProcessingResult):
    st.subheader("📜 Full English Transcript")
    transcript_text = result.translated_transcript or result.transcript
    if transcript_text:
        st.text_area(
            "Transcript Content",
            value=transcript_text,
            height=400,
            disabled=True,
            label_visibility="collapsed",
        )
        st.download_button(
            label="📥 Download Transcript (.txt)",
            data=export_transcript_txt(transcript_text),
            file_name=f"{result.video_id}_transcript.txt",
            mime="text/plain",
        )
    else:
        st.info("No transcript available.")


def render_summary_tab(result: VideoProcessingResult):
    st.subheader("📝 Structured Summary")
    if result.summary and result.summary.strip():
        st.markdown(result.summary)
        st.markdown("")
        st.download_button(
            label="📥 Download Summary (.txt)",
            data=export_summary_txt(result.summary),
            file_name=f"{result.video_id}_summary.txt",
            mime="text/plain",
        )
    else:
        st.info("No summary available for this video.")


def render_action_items_tab(result: VideoProcessingResult):
    st.subheader("✅ Action Items")
    if result.action_items:
        formatted = [
            {
                "Task": item.get("task", "Not specified"),
                "Owner": item.get("owner", "Not specified"),
                "Deadline": item.get("deadline", "Not specified"),
            }
            for item in result.action_items
        ]
        st.dataframe(formatted, use_container_width=True)
    else:
        st.info("No action items were identified in the transcript.")


def render_decisions_tab(result: VideoProcessingResult):
    st.subheader("🎯 Key Decisions")
    if result.decisions:
        for idx, item in enumerate(result.decisions, 1):
            with st.container():
                st.markdown(f"**{idx}. {item.get('decision', 'Decision')}**")
                if item.get("context"):
                    st.caption(f"*Context:* {item.get('context')}")
                st.markdown("---")
    else:
        st.info("No decisions were identified in the transcript.")


def render_questions_tab(result: VideoProcessingResult):
    st.subheader("❓ Open Questions")
    if result.open_questions:
        for idx, item in enumerate(result.open_questions, 1):
            with st.container():
                st.markdown(f"**{idx}. {item.get('question', 'Question')}**")
                if item.get("context"):
                    st.caption(f"*Context:* {item.get('context')}")
                st.markdown("---")
    else:
        st.info("No open questions were identified in the transcript.")


def render_rag_chat_tab(result: VideoProcessingResult):
    st.subheader("💬 Ask the Video")
    st.caption("Ask questions strictly answered from the video context using ChromaDB and Mistral.")

    if not result.rag_engine:
        st.warning("RAG Engine is not initialized for this video.")
        return

    # Display Chat History
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])
            if chat.get("sources"):
                with st.expander("🔍 View Retrieved Context Chunks", expanded=False):
                    for src in chat["sources"]:
                        cid = src.get("chunk_id", "?")
                        st.markdown(f"**Chunk {cid}:**\n> {src.get('content', '')}")

    # Chat Input
    if user_prompt := st.chat_input("Ask a question about this video..."):
        # Display user message immediately
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Generate Assistant Answer
        with st.chat_message("assistant"):
            with st.spinner("Searching transcript..."):
                try:
                    rag_response = result.rag_engine.ask_with_sources(user_prompt)
                    answer = rag_response.get("answer", "No answer produced.")
                    sources = rag_response.get("sources", [])

                    st.markdown(answer)
                    if sources:
                        with st.expander("🔍 View Retrieved Context Chunks", expanded=False):
                            for src in sources:
                                cid = src.get("chunk_id", "?")
                                st.markdown(f"**Chunk {cid}:**\n> {src.get('content', '')}")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                except Exception as e:
                    err = f"Failed to retrieve response: {str(e)}"
                    st.error(err)
                    st.session_state.chat_history.append({"role": "assistant", "content": err})


def render_export_tab(result: VideoProcessingResult):
    st.subheader("📦 Export Options")
    st.markdown("Download generated knowledge artifacts in text and PDF formats.")

    col1, col2, col3 = st.columns(3)
    transcript_text = result.translated_transcript or result.transcript

    with col1:
        st.markdown("##### 📜 Transcript (TXT)")
        st.caption("Raw and translated English transcript.")
        st.download_button(
            label="Download Transcript TXT",
            data=export_transcript_txt(transcript_text),
            file_name=f"{result.video_id}_transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col2:
        st.markdown("##### 📝 Summary (TXT)")
        st.caption("Structured overview and key takeaways.")
        st.download_button(
            label="Download Summary TXT",
            data=export_summary_txt(result.summary),
            file_name=f"{result.video_id}_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col3:
        st.markdown("##### 📑 Full Report (PDF)")
        st.caption("Complete meeting intelligence document.")
        pdf_cache_key = f"pdf_{result.video_id}"
        if pdf_cache_key not in st.session_state:
            st.session_state[pdf_cache_key] = export_full_report_pdf(
                title=result.title,
                summary=result.summary,
                action_items=result.action_items,
                decisions=result.decisions,
                open_questions=result.open_questions,
                source=result.source,
            )

        st.download_button(
            label="Download Full Report PDF",
            data=st.session_state[pdf_cache_key],
            file_name=f"{result.video_id}_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


def main():
    render_sidebar()
    render_header()

    if st.session_state.result is None:
        process_clicked, source_path = render_input_section()
        if process_clicked and source_path:
            execute_pipeline(source_path)
    else:
        result: VideoProcessingResult = st.session_state.result

        tabs = st.tabs([
            "📌 Overview",
            "📜 Transcript",
            "📝 Summary",
            "✅ Action Items",
            "🎯 Decisions",
            "❓ Open Questions",
            "💬 Ask the Video",
            "📦 Export",
        ])

        with tabs[0]:
            render_overview_tab(result)
        with tabs[1]:
            render_transcript_tab(result)
        with tabs[2]:
            render_summary_tab(result)
        with tabs[3]:
            render_action_items_tab(result)
        with tabs[4]:
            render_decisions_tab(result)
        with tabs[5]:
            render_questions_tab(result)
        with tabs[6]:
            render_rag_chat_tab(result)
        with tabs[7]:
            render_export_tab(result)


if __name__ == "__main__":
    main()
