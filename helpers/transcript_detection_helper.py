"""Helpers for detecting transcript availability on YouTube videos."""

import logging
from typing import Any

import yt_dlp
from yt_dlp.utils import DownloadError

from helpers.downloading_transcript_helper import get_english_transcript_tracks

from helpers.shared_helper import setup_logging

MINIMUM_TRANSCRIPT_DURATION_SECONDS = 15 * 60


def parse_duration_to_seconds(duration_text: str) -> int:
    """
    Convert a hh:mm:ss duration string to seconds.

    Arguments:
        duration_text (str): Duration formatted as hh:mm:ss.

    Returns:
        int: Total duration in seconds.
    """
    hours_text, minutes_text, seconds_text = duration_text.split(":")
    hours = int(hours_text)
    minutes = int(minutes_text)
    seconds = int(seconds_text)

    total_seconds = hours * 3600 + minutes * 60 + seconds

    return total_seconds


def is_transcript_eligible(record: dict[str, Any]) -> bool:
    """
    Check whether a detected video should be sent to transcript detection.

    Arguments:
        record (dict[str, Any]): Detection record from the video table.

    Returns:
        bool: True when the video is longer than the minimum duration.
    """
    duration_text = str(record.get("duration", "00:00:00"))
    duration_seconds = parse_duration_to_seconds(duration_text)

    is_eligible = duration_seconds >= MINIMUM_TRANSCRIPT_DURATION_SECONDS

    return is_eligible


def detect_transcript_data(video_url: str) -> dict[str, Any]:
    """
    Detect transcript tracks and availability for one YouTube video.

    Arguments:
        video_url (str): Full YouTube video URL.

    Returns:
        dict[str, Any]: Transcript metadata including availability, languages,
            track count, track URLs, and any error string.

    Example:
        >>> data = detect_transcript_data("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        >>> isinstance(data["transcript_available"], bool)
        True
    """
    ydl_opts = {"skip_download": True, "quiet": True, "no_warnings": True}

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
    except DownloadError as exc:
        transcript_data = {
            "transcript_available": False,
            "transcript_track_count": 0,
            "transcript_languages": [],
            "transcript_urls": [],
            "transcript_error": str(exc),
        }

        return transcript_data

    tracks = get_english_transcript_tracks(info)
    languages = []
    transcript_urls = []

    for track in tracks:
        language = track.get("language") or track.get("lang") or "en"
        languages.append(language)

        url = track.get("url")
        if url:
            transcript_urls.append(url)

    transcript_data = {
        "transcript_available": bool(tracks),
        "transcript_track_count": len(tracks),
        "transcript_languages": languages,
        "transcript_urls": transcript_urls,
        "transcript_error": None,
    }

    return transcript_data


def detect_transcripts_for_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Enrich detection records with transcript availability information.

    Arguments:
        records (list[dict[str, Any]]): Flat records from the video detection stage.

    Returns:
        list[dict[str, Any]]: The same records with transcript metadata added.

    Example:
        >>> detect_transcripts_for_records([])
        []
    """
    enriched_records: list[dict[str, Any]] = []

    for record in records:
        if not is_transcript_eligible(record):
            continue

        transcript_data = detect_transcript_data(record["url"])
        enriched_record = dict(record)
        enriched_record.update(transcript_data)
        enriched_records.append(enriched_record)

        if transcript_data["transcript_error"]:
            logging.warning(
                "Transcript detection failed for %s: %s",
                record.get("title", record["url"]),
                transcript_data["transcript_error"],
            )

    result_records = enriched_records

    return result_records