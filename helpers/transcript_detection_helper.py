"""Helpers for detecting transcript availability on YouTube videos."""

from typing import Any

import yt_dlp

from helpers.downloading_transcript_helper import get_english_transcript_tracks

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