import re
import uuid
from urllib.parse import parse_qs, urlparse

import yt_dlp  # Used to download YouTube audio
from faster_whisper import WhisperModel  # Efficient transcription model

from configs.config import Config


def extract_youtube_id(url: str) -> str | None:
    """Extract the YouTube video ID from different URL formats."""
    parsed_url = urlparse(url)

    # Standard YouTube URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID)
    if parsed_url.hostname in {"www.youtube.com", "youtube.com"}:
        query = parse_qs(parsed_url.query)
        return query.get("v", [None])[0]

    # Shortened YouTube URL (e.g., https://youtu.be/VIDEO_ID)
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.strip("/")

    # Embedded video URL (e.g., https://www.youtube.com/embed/VIDEO_ID)
    match = re.match(r"^/embed/([a-zA-Z0-9_-]{11})", parsed_url.path)
    if match:
        return match.group(1)

    return None


def transcribe_youtube_audio(
    youtube_url: str,
    video_name: str,
    model_size: str = "base",
    device: str = "cpu",
) -> str | None:
    """Download audio from a YouTube video, transcribe it using Whisper, and save the transcript."""
    # Ensure audio output directory exists
    audio_dir = Config.paths.YOUTUBE_DIR / "audios"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Create unique filename
    file_prefix = audio_dir / f"{video_name}_{uuid.uuid4().hex}"
    output_template = str(file_prefix) + ".%(ext)s"

    print(f"Downloading audio from YouTube: {youtube_url}")

    # yt-dlp options for downloading audio
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "retries": 3,
        "socket_timeout": 20,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        "cookies_from_browser": ("chrome",),
        "nocheckcertificate": True,
    }

    # Download audio safely
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
    except yt_dlp.utils.DownloadError as e:
        print(f"yt-dlp Download Error: {e}")
        return None
    except Exception as exc:
        print(f"Unexpected error during download: {exc}")
        return None

    # Locate downloaded file
    mp3_path = file_prefix.with_suffix(".mp3")
    if not mp3_path.exists() or mp3_path.stat().st_size == 0:
        print("Downloaded audio file is empty or missing.")
        return None

    print(f"Audio saved to: {mp3_path} ({mp3_path.stat().st_size} bytes)")

    # Transcribe audio
    print("Transcribing audio...")
    model = WhisperModel(model_size, device=device)
    segments, _ = model.transcribe(str(mp3_path), beam_size=5)
    text = " ".join(segment.text for segment in segments)

    # Save transcript safely
    yb_text_path = Config.paths.YOUTUBE_DIR / f"yb_{video_name}.txt"
    with yb_text_path.open("w", encoding="utf-8") as f:  # PTH123
        f.write(text)

    print(f"Transcript saved to: {yb_text_path}")
    return text
