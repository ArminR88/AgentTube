"""Helpers for downloading plain English transcripts from YouTube videos."""

import logging
import re
import time
from pathlib import Path
from typing import Any

from requests import HTTPError
import yt_dlp

from helpers.shared_helper import dev_request_json

TIMEOUT = 30
INTER_VIDEO_DELAY_SECONDS = 20


def sanitize_filename_part(text: str) -> str:
    """
    Convert text into a safe filename fragment.

    Arguments:
        text (str): Input text.

    Returns:
        str: Sanitized filename fragment.

    Example:
        >>> sanitize_filename_part("Judge Napolitano!")
        'Judge_Napolitano'
    """
    cleaned_text = re.sub(r"[^0-9A-Za-z_-]+", "_", text.strip())
    normalized_text = re.sub(r"_+", "_", cleaned_text).strip("_")

    if not normalized_text:
        normalized_text = "unknown"

    return normalized_text


def build_transcript_filename(channel_name: str | None, video_id: str) -> str:
    """
    Build a transcript filename from a channel name and video ID.

    Arguments:
        channel_name (str | None): Channel name used as a prefix.
        video_id (str): YouTube video ID.

    Returns:
        str: Filename ending in .txt.

    Example:
        >>> build_transcript_filename("Judge Napolitano", "dQw4w9WgXcQ")
        'Judge_Napolitano_dQw4w9WgXcQ.txt'
    """
    has_channel_name = bool(channel_name)
    if has_channel_name == True:
        prefix = sanitize_filename_part(channel_name)
    else:
        prefix = "unknown"
    filename = f"{prefix}_{video_id}.txt"

    return filename


def get_video_id(url: str) -> str:
    """
    Extract YouTube video ID from URL.

    Arguments:
        url (str): YouTube video URL.

    Returns:
        str: 11-character video ID, or "unknown" if extraction fails.

    Example:
        >>> get_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
    """
    match = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})(?:[?&]|$)", url)
    if match:
        video_id = match.group(1)
    else:
        video_id = "unknown"

    return video_id


def get_english_transcript_tracks(info: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract English transcript tracks from YouTube video metadata.

    Arguments:
        info (dict[str, Any]): Video metadata dictionary from yt-dlp.

    Returns:
        list[dict[str, Any]]: Transcript track dictionaries, or an empty list.
    """
    captions = info.get("automatic_captions") or {}
    subtitles = info.get("subtitles") or {}
    caption_tracks = captions.get("en")
    subtitle_tracks = subtitles.get("en")
    if caption_tracks:
        transcript_tracks = caption_tracks
    elif subtitle_tracks:
        transcript_tracks = subtitle_tracks
    else:
        transcript_tracks = []

    return transcript_tracks


def fetch_plain_text(transcript_url: str) -> str:
    """
    Fetch and flatten YouTube transcript JSON into plain text.

    Arguments:
        transcript_url (str): URL to the YouTube transcript JSON endpoint.

    Returns:
        str: Plain text transcript with normalized whitespace.
    """
    data = dev_request_json(transcript_url, timeout=TIMEOUT)

    text_parts = []
    for event in data.get("events", []):
        for seg in event.get("segs", []):
            text = seg.get("utf8")
            if text:
                text_parts.append(text)

    plain_text = " ".join(text_parts)
    normalized_plain_text = " ".join(plain_text.split())

    return normalized_plain_text


def download_transcript(video_url: str, output_dir: str = "transcripts", channel_name: str | None = None) -> bool:
    """
    Download plain English transcript for a single YouTube video.

    Arguments:
        video_url (str): Full YouTube video URL.
        output_dir (str): Directory to save transcript files.

    Returns:
        bool: True if transcript was downloaded, False otherwise.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ydl_opts = {"skip_download": True, "quiet": True, "no_warnings": True}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)

    transcript_tracks = get_english_transcript_tracks(info)
    has_no_transcript_tracks = len(transcript_tracks) == 0
    if has_no_transcript_tracks == True:
        logging.warning("No English transcript available for %s", video_url)
        is_downloaded = False
        return is_downloaded

    first_track = transcript_tracks[0]
    first_track_url = first_track["url"]
    plain_text = fetch_plain_text(first_track_url)

    video_id = get_video_id(video_url)
    output_file = Path(output_dir) / build_transcript_filename(channel_name, video_id)

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(plain_text)

    logging.info("Downloaded: %s (%s chars)", output_file.name, len(plain_text))
    is_downloaded = True
    return is_downloaded


def download_transcript_from_record(record: dict[str, Any], output_dir: str = "transcripts") -> bool:
    """
    Download a transcript using a record enriched by transcript detection.

    Arguments:
        record (dict[str, Any]): A detection record with transcript metadata.
        output_dir (str): Directory to save transcript files.

    Returns:
        bool: True if transcript was downloaded, False otherwise.

    Example:
        >>> download_transcript_from_record({"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "transcript_available": False})
        False
    """
    transcript_is_available = bool(record.get("transcript_available", True))
    if transcript_is_available == False:
        logging.warning("Skipping video without transcript: %s", record.get("title", record["url"]))
        is_downloaded = False
        return is_downloaded

    transcript_urls = record.get("transcript_urls") or []
    is_downloaded = False

    has_transcript_urls = len(transcript_urls) > 0
    if has_transcript_urls == True:
        transcript_url = transcript_urls[0]
        plain_text = fetch_plain_text(transcript_url)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        video_id = get_video_id(record["url"])
        output_file = Path(output_dir) / build_transcript_filename(record.get("channel_name"), video_id)

        with open(output_file, "w", encoding="utf-8") as file:
            file.write(plain_text)

        logging.info("Downloaded: %s (%s chars)", output_file.name, len(plain_text))
        is_downloaded = True
    else:
        is_downloaded = download_transcript(record["url"], output_dir, channel_name=record.get("channel_name"))

    return is_downloaded


def download_transcripts_from_records(
    records: list[dict[str, Any]],
    output_dir: str = "transcripts",
    inter_video_delay: int | None = None,
) -> dict[str, int]:
    """
    Download transcripts for detection records.

    Arguments:
        records (list[dict[str, Any]]): Detection records from the pipeline.
        output_dir (str): Directory to save transcript files.
        inter_video_delay (int | None): Seconds to sleep between videos.
            Uses INTER_VIDEO_DELAY_SECONDS when not provided.

    Returns:
        dict[str, int]: Statistics with keys 'success', 'failed', and 'total'.

    Notes:
        Stops the batch on the first 429 rate-limit response to avoid extra requests.
    """
    stats = {
        "success": 0,
        "failed": 0,
        "total": len(records),
    }
    has_custom_delay = inter_video_delay is not None
    if has_custom_delay == True:
        delay = inter_video_delay
    else:
        delay = INTER_VIDEO_DELAY_SECONDS

    for index, record in enumerate(records, 1):
        logging.info("Processing %s/%s: %s", index, stats["total"], record["url"])
        try:
            transcript_downloaded = download_transcript_from_record(record, output_dir)
        except HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            if status_code == 429:
                logging.warning("Rate limited on %s; stopping batch", record["url"])
                break
            logging.warning("Transcript request failed for %s: %s", record["url"], exc)
            transcript_downloaded = False
        if transcript_downloaded == True:
            stats["success"] += 1
        else:
            stats["failed"] += 1

        logging.debug("Sleeping %s seconds before next video", delay)
        time.sleep(delay)

    return stats