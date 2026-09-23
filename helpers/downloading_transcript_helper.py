"""Helpers for downloading plain English transcripts from YouTube videos."""

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from requests import RequestException
import yt_dlp
from yt_dlp.utils import DownloadError

from helpers.shared_helper import DevTooManyAttemptsError, dev_request_json, setup_logging

TIMEOUT = 30
DEV_TRANSCRIPT_RATE_LIMIT_LIMIT = 3
DEV_TRANSCRIPT_RATE_LIMIT_STREAK = 0


def is_rate_limit_error(exc: Exception) -> bool:
    """
    Check whether an exception looks like a YouTube rate-limit response.

    Arguments:
        exc (Exception): Exception raised while fetching transcript data.

    Returns:
        bool: True when the error is a 429 / too-many-requests response.
    """
    error_text = str(exc)
    rate_limited = "429" in error_text or "Too Many Requests" in error_text

    return rate_limited


def dev_reset_transcript_rate_limit_streak() -> None:
    """
    Reset the dev-only transcript rate-limit streak.

    Arguments:
        None

    Returns:
        None
    """
    global DEV_TRANSCRIPT_RATE_LIMIT_STREAK
    DEV_TRANSCRIPT_RATE_LIMIT_STREAK = 0


def dev_handle_transcript_rate_limit(video_url: str, exc: RequestException) -> None:
    """
    Log transcript rate limits and apply cooldowns without aborting the batch.

    Arguments:
        video_url (str): YouTube video URL.
        exc (RequestException): Error raised while fetching transcript data.

    Returns:
        None

    Example:
        >>> dev_handle_transcript_rate_limit("https://www.youtube.com/watch?v=dQw4w9WgXcQ", RequestException("429"))
        >>> True
        True
    """
    global DEV_TRANSCRIPT_RATE_LIMIT_STREAK

    DEV_TRANSCRIPT_RATE_LIMIT_STREAK += 1
    current_streak = DEV_TRANSCRIPT_RATE_LIMIT_STREAK

    logging.warning(
        "Transcript fetch rate-limited for %s (%s/%s): %s",
        video_url,
        current_streak,
        DEV_TRANSCRIPT_RATE_LIMIT_LIMIT,
        exc,
    )

    if current_streak == 1:
        cooldown_seconds = 0
    elif current_streak == 2:
        cooldown_seconds = 5
    elif current_streak == 3:
        cooldown_seconds = 15
    else:
        cooldown_seconds = 30

    if current_streak >= DEV_TRANSCRIPT_RATE_LIMIT_LIMIT:
        logging.warning(
            "Dev transcript attempt limit reached at %s consecutive rate limits for %s; continuing batch but subsequent videos may also be rate-limited",
            current_streak,
            video_url,
        )

    if cooldown_seconds > 0:
        time.sleep(cooldown_seconds)


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
    prefix = sanitize_filename_part(channel_name) if channel_name else "unknown"
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
    transcript_tracks = captions.get("en") or subtitles.get("en") or []

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

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
    except DownloadError as exc:
        logging.error("Failed to read YouTube metadata for %s: %s", video_url, exc)
        failed = False

        return failed

    transcript_tracks = get_english_transcript_tracks(info)
    if not transcript_tracks:
        logging.warning("No English transcript available for %s", video_url)
        failed = False

        return failed

    try:
        plain_text = fetch_plain_text(transcript_tracks[0]["url"])
        dev_reset_transcript_rate_limit_streak()
    except RequestException as exc:
        dev_handle_transcript_rate_limit(video_url, exc)
        failed = False

        return failed
    except ValueError as exc:
        logging.error("Invalid transcript JSON for %s: %s", video_url, exc)
        failed = False

        return failed

    video_id = get_video_id(video_url)
    output_file = Path(output_dir) / build_transcript_filename(channel_name, video_id)

    try:
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(plain_text)
    except OSError as exc:
        logging.error("Failed to write transcript file %s: %s", output_file, exc)
        failed = False

        return failed

    logging.info("Downloaded: %s (%s chars)", output_file.name, len(plain_text))
    downloaded = True

    return downloaded


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
    if not record.get("transcript_available", True):
        logging.warning("Skipping video without transcript: %s", record.get("title", record["url"]))
        skipped = False

        return skipped

    transcript_urls = record.get("transcript_urls") or []
    downloaded = False

    if transcript_urls:
        try:
            plain_text = fetch_plain_text(transcript_urls[0])
            dev_reset_transcript_rate_limit_streak()
        except RequestException as exc:
            dev_handle_transcript_rate_limit(record["url"], exc)
            failed = False

            return failed
        except ValueError as exc:
            logging.error("Invalid transcript JSON for %s: %s", record["url"], exc)
            failed = False

            return failed

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        video_id = get_video_id(record["url"])
        output_file = Path(output_dir) / build_transcript_filename(record.get("channel_name"), video_id)

        try:
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(plain_text)
        except OSError as exc:
            logging.error("Failed to write transcript file %s: %s", output_file, exc)
            failed = False

            return failed

        logging.info("Downloaded: %s (%s chars)", output_file.name, len(plain_text))
        downloaded = True
    else:
        downloaded = download_transcript(record["url"], output_dir, channel_name=record.get("channel_name"))

    return downloaded


def download_transcripts(video_urls: list[str], output_dir: str = "transcripts") -> dict[str, int]:
    """
    Download transcripts for multiple YouTube videos.

    Arguments:
        video_urls (list[str]): List of YouTube video URLs.
        output_dir (str): Directory to save transcript files.

    Returns:
        dict[str, int]: Statistics with keys 'success', 'failed', and 'total'.

    Notes:
        Videos that fail due to rate limits are skipped and the batch continues.
    """
    stats = {
        "success": 0,
        "failed": 0,
        "total": len(video_urls),
        "dev_rate_limit_exhausted": False,
        "dev_rate_limit_error": None,
    }

    for index, url in enumerate(video_urls, 1):
        logging.info("Processing %s/%s: %s", index, stats["total"], url)
        if download_transcript(url, output_dir):
            stats["success"] += 1
        else:
            stats["failed"] += 1

    if stats["failed"] > 0:
        logging.warning(
            "Download batch finished with %s failed / %s total videos",
            stats["failed"],
            stats["total"],
        )

    result_stats = stats

    return result_stats


def download_transcripts_from_records(
    records: list[dict[str, Any]],
    output_dir: str = "transcripts",
) -> dict[str, int]:
    """
    Download transcripts for detection records.

    Arguments:
        records (list[dict[str, Any]]): Detection records from the pipeline.
        output_dir (str): Directory to save transcript files.

    Returns:
        dict[str, int]: Statistics with keys 'success', 'failed', and 'total'.

    Notes:
        Videos that fail due to rate limits are skipped and the batch continues.
    """
    stats = {
        "success": 0,
        "failed": 0,
        "total": len(records),
        "dev_rate_limit_exhausted": False,
        "dev_rate_limit_error": None,
    }

    for index, record in enumerate(records, 1):
        logging.info("Processing %s/%s: %s", index, stats["total"], record["url"])
        if download_transcript_from_record(record, output_dir):
            stats["success"] += 1
        else:
            stats["failed"] += 1

    if stats["failed"] > 0:
        logging.warning(
            "Download batch finished with %s failed / %s total videos",
            stats["failed"],
            stats["total"],
        )

    result_stats = stats

    return result_stats


def get_urls_from_args(args: Any) -> list[str]:
    """
    Extract video URLs from command-line arguments.

    Arguments:
        args: Parsed argparse namespace containing 'urls' and 'file' attributes.

    Returns:
        list[str]: List of video URLs from command line and/or file.
    """
    video_urls = list(args.urls)

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as file:
                file_urls = [line.strip() for line in file if line.strip()]
                video_urls.extend(file_urls)
        except OSError as exc:
            logging.error("Failed to read URL file: %s", exc)
            sys.exit(1)

    result_urls = video_urls

    return result_urls


def parse_arguments() -> Any:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Download YouTube transcripts in batch")
    parser.add_argument("urls", nargs="*", help="YouTube video URLs to download")
    parser.add_argument("-f", "--file", help="File containing list of URLs (one per line)")
    parser.add_argument("-o", "--output", default="transcripts", help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parsed_arguments = parser.parse_args()

    return parsed_arguments


def main() -> None:
    """
    Main entry point for batch transcript downloading.

    Returns:
        None
    """
    args = parse_arguments()
    setup_logging(args.verbose)

    video_urls = get_urls_from_args(args)
    if not video_urls:
        video_urls = ["https://www.youtube.com/watch?v=OkhHCr83mko"]
        logging.info("No URLs provided, using default test video")

    stats = download_transcripts(video_urls, args.output)

    logging.info("%s", "=" * 50)
    logging.info(
        "SUMMARY: %s successful, %s failed, %s total",
        stats["success"],
        stats["failed"],
        stats["total"],
    )

    if stats["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()