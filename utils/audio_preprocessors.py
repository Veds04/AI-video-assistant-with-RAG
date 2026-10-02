import os
import yt_dlp
from pydub import AudioSegment
from rich import print
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn


console = Console()

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_video(url: str) -> str:
    """Download video/audio from a supported URL and return its path."""

    opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = ydl.prepare_filename(info)

            # yt-dlp may merge the final file into MP4
            if not os.path.exists(path):
                mp4_path = os.path.splitext(path)[0] + ".mp4"
                if os.path.exists(mp4_path):
                    path = mp4_path

        if not os.path.exists(path):
            raise FileNotFoundError("Downloaded file was not found.")

        return os.path.abspath(path)

    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")


def convert_to_wav(input_path: str) -> str:
    """Convert media to 16kHz mono WAV."""

    try:
        output_path = os.path.splitext(input_path)[0] + "_converted.wav"

        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")

        return output_path

    except Exception as e:
        raise RuntimeError(f"Audio extraction failed: {e}")


def split_audio(input_path: str, chunk_minutes: int = 10) -> list[str]:
    """Split WAV audio into chunks and return their paths."""

    try:
        audio = AudioSegment.from_file(input_path)
        chunk_ms = chunk_minutes * 60 * 1000

        output_dir = os.path.splitext(input_path)[0] + "_chunks"
        os.makedirs(output_dir, exist_ok=True)

        chunks = []

        for i, start in enumerate(range(0, len(audio), chunk_ms), 1):
            chunk = audio[start:start + chunk_ms]

            path = os.path.join(
                output_dir,
                f"chunk_{i:03d}.wav"
            )

            chunk.export(path, format="wav")
            chunks.append(path)

        return chunks

    except Exception as e:
        raise RuntimeError(f"Chunking failed: {e}")


def process_video(url: str):
    try:
        with console.status(
            "[bold cyan]Downloading video...[/bold cyan]",
            spinner="dots"
        ):
            video_path = download_video(url)

        print(f"[green]✓[/green] Download complete")

        with console.status(
            "[bold yellow]Extracting audio...[/bold yellow]",
            spinner="dots"
        ):
            wav_path = convert_to_wav(video_path)

        print("[green]✓[/green] Audio extracted")

        with console.status(
            "[bold magenta]Creating audio chunks...[/bold magenta]",
            spinner="dots"
        ):
            chunks = split_audio(wav_path, 10)

        print(f"[green]✓[/green] Created {len(chunks)} chunks")

        print("\n[bold green]✓ Processing complete[/bold green]\n")

        for i, chunk in enumerate(chunks, 1):
            print(f"[cyan]{i:02}[/cyan] → {chunk}")

        return chunks

    except Exception as e:
        print(f"[bold red]✗ Error:[/bold red] {e}")
        return []


if __name__ == "__main__":

    print("\n[bold cyan]AI Video Assistant with RAG[/bold cyan]")
    print("─" * 40)

    url = input("\nPaste video URL: ").strip()

    if not url:
        print("[red]Please enter a valid URL.[/red]")
    else:
        process_video(url)