"""Helpers for detecting transcript availability on YouTube videos."""

import logging
from typing import Any

import yt_dlp

from helpers.downloading_transcript_helper import get_english_transcript_tracks
from helpers.detecting_videos_helper import CHANNEL_NAMES


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

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)

    tracks = get_english_transcript_tracks(info)
    languages = []
    transcript_urls = []

    for track in tracks:
        language = track.get("language")
        if language is None:
            language = track.get("lang")
        if language is None:
            language = "en"
        languages.append(language)

        url = track.get("url")
        has_url = bool(url)
        if has_url == True:
            transcript_urls.append(url)

    transcript_available = bool(tracks)
    transcript_data = {
        "transcript_available": transcript_available,
        "transcript_track_count": len(tracks),
        "transcript_languages": languages,
        "transcript_urls": transcript_urls,
        "transcript_error": "",
    }

    return transcript_data


def format_elapsed_seconds(seconds: float) -> str:
    """
    Format an elapsed duration as seconds or minutes+seconds.

    Arguments:
        seconds (float): Elapsed time in seconds.

    Returns:
        str: Formatted duration, e.g. "4.2s" or "1m 5.2s".

    Example:
        >>> format_elapsed_seconds(4.2)
        '4.2s'
        >>> format_elapsed_seconds(65.2)
        '1m 5.2s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}m {remainder:.1f}s"


def print_transcript_detection_header() -> None:
    """
    Print the deterministic header of the stage 2 banner.

    Arguments:
        None

    Returns:
        None

    Example:
        >>> print_transcript_detection_header()
        ----------------------------------------------------------------
        ################# Stage 2: Transcript Detection #################
        ----------------------------------------------------------------
        Channel                        |    All | With Transcript
        ----------------------------------------------------------------
    """
    print("-" * 64)
    print("################# Stage 2: Transcript Detection #################")
    print("-" * 64)
    print(f"{'Channel':<30} | {'All':>6} | {'With Transcript':>15}")
    print("-" * 64)


def print_transcript_detection_body(enriched_records: list[dict[str, Any]]) -> None:
    """
    Print the per-channel counts of transcript availability for stage 2.

    Arguments:
        enriched_records (list[dict[str, Any]]): Records already enriched
            with transcript metadata by detect_transcripts_for_records().

    Returns:
        None

    Example:
        >>> print_transcript_detection_body([])
        ----------------------------------------------------------------
        Total                          |      0 |               0
        ----------------------------------------------------------------
    """
    channel_counts = {}

    for channel_name in CHANNEL_NAMES.values():
        channel_counts[channel_name] = {"all": 0, "with_transcript": 0}

    for record in enriched_records:
        channel_name = record.get("channel_name", "")
        if channel_name not in channel_counts:
            channel_counts[channel_name] = {"all": 0, "with_transcript": 0}

        channel_counts[channel_name]["all"] += 1

        transcript_available = record.get("transcript_available", False)
        if transcript_available == True:
            channel_counts[channel_name]["with_transcript"] += 1

    total_all = 0
    total_with_transcript = 0

    for channel_name, counts in channel_counts.items():
        all_count = counts["all"]
        with_transcript_count = counts["with_transcript"]
        total_all += all_count
        total_with_transcript += with_transcript_count
        print(f"{channel_name:<30} | {all_count:>6} | {with_transcript_count:>15}")

    print("-" * 64)
    print(f"{'Total':<30} | {total_all:>6} | {total_with_transcript:>15}")
    print("-" * 64)


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
        transcript_data = detect_transcript_data(record["url"])
        enriched_record = dict(record)
        enriched_record.update(transcript_data)
        enriched_records.append(enriched_record)

    return enriched_records