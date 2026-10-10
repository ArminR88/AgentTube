"""Helpers for detecting recent YouTube videos and shaping the detection schema."""

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from requests import RequestException

from helpers.shared_helper import dev_request_json

API_KEY_ENV = "YOUTUBE_API_KEY"
START_HOUR = 17
LOCAL_TIMEZONE = ZoneInfo("Europe/Berlin")
TIMEOUT_SECONDS = 30
MAX_RESULTS = 50
MINIMUM_TRANSCRIPT_DURATION_SECONDS = 18 * 60
VIDEO_DURATION_PATTERN = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)

CHANNEL_IDS = [
    "UCOxLhz6B_elvLflntSEfnzA",  # Danny_Haiphong
    "UCDkEYb-TXJVWLvOokshtlsw",  # Judge_Napolitano
    "UCZFCDIHTe9HGxtIuVDpBz7g",  # Glenn Diesen
    "UCWDN5zr5ttctoIAhZwW6tcQ",  # Daniel Davis / Deep Dive
    "UCTWBp-39z6tvz4-LQB-Z_QA",  # Mario_Nawfal
    "UCEATT6H3U5lu20eKPuHVN8A",  # Chris_Hedges
    "UCewRbK22LRnNi6N3EcGjbow",  # Transition_Protocol
    "UCkF-6h_Zgf9zXNUmUB-MzTw",  # Dialogue_Works
]

CHANNEL_NAMES = {
    "UCOxLhz6B_elvLflntSEfnzA": "Danny_Haiphong",
    "UCDkEYb-TXJVWLvOokshtlsw": "Judge_Napolitano",
    "UCZFCDIHTe9HGxtIuVDpBz7g": "Glenn_Diesen",
    "UCWDN5zr5ttctoIAhZwW6tcQ": "Daniel_Davis",
    "UCTWBp-39z6tvz4-LQB-Z_QA": "Mario_Nawfal",
    "UCEATT6H3U5lu20eKPuHVN8A": "Chris_Hedges",
    "UCewRbK22LRnNi6N3EcGjbow": "Transition_Protocol",
    "UCkF-6h_Zgf9zXNUmUB-MzTw": "Dialogue_Works_Nima",
}

CHANNEL_TITLE_FILTERS = {
    "UCTWBp-39z6tvz4-LQB-Z_QA": ["parsi", "johnson", "pape", "mearsheimer", "wilkerson", "escobar", "diesen", "hudson"],
}


def get_api_key() -> str | None:
    """
    Read the YouTube API key from the environment.

    Arguments:
        None

    Returns:
        str | None: API key value or None if not set.

    Example:
        >>> key = get_api_key()
        >>> if key:
        ...     print("API key found")
    """
    api_key = os.environ.get(API_KEY_ENV)
    return api_key


def get_time_window() -> tuple[str, str]:
    """
    Calculate the fixed 24-hour window from yesterday 17:00 to today 17:00 local time.

    Arguments:
        None

    Returns:
        tuple[str, str]: (published_after, published_before) in UTC ISO format.

    Example:
        >>> after, before = get_time_window()
        >>> after.endswith('Z')
        True
    """
    now = datetime.now(LOCAL_TIMEZONE)
    end = now.replace(hour=START_HOUR, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=1)
    # Keep a fixed 24h window anchored at START_HOUR in local time.
    end = start + timedelta(days=1)
    published_after = start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    published_before = end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    time_window = (published_after, published_before)
    return time_window


def get_channel_name(channel_id: str) -> str:
    """
    Return the display name for a channel ID.

    Arguments:
        channel_id (str): YouTube channel ID.

    Returns:
        str: Channel name or original ID if not found.

    Example:
        >>> get_channel_name("UCOxLhz6B_elvLflntSEfnzA")
        'Danny_Haiphong'
    """
    channel_name = CHANNEL_NAMES.get(channel_id, channel_id)
    return channel_name


def fetch_channel_videos(
    channel_id: str,
    published_after: str,
    published_before: str,
    api_key: str,
) -> list[dict[str, Any]]:
    """
    Fetch recent videos for one channel using the YouTube API.

    Arguments:
        channel_id (str): YouTube channel ID.
        published_after (str): UTC timestamp for window start.
        published_before (str): UTC timestamp for window end.
        api_key (str): YouTube API key.

    Returns:
        list[dict[str, Any]]: List of non-live video metadata objects.

    Example:
        >>> videos = fetch_channel_videos(
        ...     "UCOxLhz6B_elvLflntSEfnzA",
        ...     "2024-01-15T16:00:00Z",
        ...     "2024-01-16T16:00:00Z",
        ...     "key",
        ... )
        >>> len(videos)
        2
    """
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "key": api_key,
        "channelId": channel_id,
        "part": "snippet",
        "order": "date",
        "maxResults": MAX_RESULTS,
        "publishedAfter": published_after,
        "publishedBefore": published_before,
        "type": "video",
    }

    # Fetch and normalize the returned search items into flat video records.
    videos: list[dict[str, Any]] = []
    for item in dev_request_json(url, timeout=TIMEOUT_SECONDS, params=params).get("items", []):
        snippet = item["snippet"]
        live_status = snippet.get("liveBroadcastContent", "none")
        is_live_or_upcoming = live_status != "none"
        # Skip live/upcoming streams; only process regular uploaded videos.
        if is_live_or_upcoming == True:
            logging.debug(
                "Skipping %s broadcast for channel %s: %s",
                live_status,
                channel_id,
                snippet.get("title", "<untitled>"),
            )
            continue

        video_id = item["id"]["videoId"]
        videos.append(
            {
                "title": snippet["title"],
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "video_id": video_id,
                "published_at": snippet["publishedAt"],
                "description": snippet.get("description", ""),
                "thumbnail": snippet["thumbnails"]["default"]["url"],
                "channel_name": snippet.get("channelTitle", ""),
            }
        )

    return videos


def fetch_video_durations(video_ids: list[str], api_key: str) -> dict[str, str]:
    """
    Fetch ISO 8601 durations for a batch of video IDs.

    Arguments:
        video_ids (list[str]): Video IDs to look up.
        api_key (str): YouTube API key.

    Returns:
        dict[str, str]: Durations keyed by video ID in hh:mm:ss format.

    Example:
        >>> durations = fetch_video_durations(["abc123"], "key")
        >>> isinstance(durations, dict)
        True
    """
    has_no_video_ids = len(video_ids) == 0
    if has_no_video_ids == True:
        empty_durations: dict[str, str] = {}
        return empty_durations

    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "key": api_key,
        "part": "contentDetails",
        "id": ",".join(video_ids),
        "maxResults": MAX_RESULTS,
    }

    durations: dict[str, str] = {}
    for item in dev_request_json(url, timeout=TIMEOUT_SECONDS, params=params).get("items", []):
        video_id = item.get("id")
        duration_text = item.get("contentDetails", {}).get("duration")
        has_duration_payload = bool(video_id) and bool(duration_text)
        if has_duration_payload == True:
            durations[video_id] = format_duration(duration_text)
    return durations


def format_duration(duration_text: str) -> str:
    """
    Convert a YouTube ISO 8601 duration to hh:mm:ss.

    Arguments:
        duration_text (str): ISO 8601 duration string.

    Returns:
        str: Zero-padded duration text.

    Example:
        >>> format_duration("PT1H2M3S")
        '01:02:03'
    """
    match = VIDEO_DURATION_PATTERN.match(duration_text)
    has_no_match = match is None
    if has_no_match == True:
        fallback_duration = "00:00:00"
        return fallback_duration

    days_text = match.group("days")
    if days_text is None:
        days = 0
    else:
        days = int(days_text)

    hours_text = match.group("hours")
    if hours_text is None:
        hours = 0
    else:
        hours = int(hours_text)

    minutes_text = match.group("minutes")
    if minutes_text is None:
        minutes = 0
    else:
        minutes = int(minutes_text)

    seconds_text = match.group("seconds")
    if seconds_text is None:
        seconds = 0
    else:
        seconds = int(seconds_text)
    total_hours = days * 24 + hours
    formatted_duration = f"{total_hours:02d}:{minutes:02d}:{seconds:02d}"
    return formatted_duration


def detect_recent_videos(api_key: str) -> dict[str, dict[str, Any]]:
    """
    Detect videos for all configured channels in the time window.

    Arguments:
        api_key (str): YouTube API key.

    Returns:
        dict[str, dict[str, Any]]: Results keyed by channel ID with name, videos, and error.

    Example:
        >>> results = detect_recent_videos("key")
        >>> "UCOxLhz6B_elvLflntSEfnzA" in results
        True
    """
    time_window = get_time_window()
    published_after = time_window[0]
    published_before = time_window[1]
    results: dict[str, dict[str, Any]] = {}

    for channel_id in CHANNEL_IDS:
        channel_name = get_channel_name(channel_id)
        logging.debug("Checking channel: %s", channel_name)
        try:
            videos = fetch_channel_videos(channel_id, published_after, published_before, api_key)
        except RequestException as exc:
            logging.error("Failed to fetch videos for %s: %s", channel_name, exc)
            results[channel_id] = {"name": channel_name, "videos": [], "error": str(exc)}
            continue

        # Build one batch duration lookup and then annotate each video in-place.
        video_ids = [video["video_id"] for video in videos]
        durations_by_id = fetch_video_durations(video_ids, api_key)
        for video in videos:
            video["duration"] = durations_by_id.get(video["video_id"], "00:00:00")

        results[channel_id] = {"name": channel_name, "videos": videos, "error": None}

    return results


def title_matches_filter(title: str, keywords: list[str]) -> bool:
    """
    Check whether a video title contains any of the keywords.

    Arguments:
        title (str): Video title.
        keywords (list[str]): Keywords to match against.

    Returns:
        bool: True if any keyword appears in the title (case-insensitive).

    Example:
        >>> title_matches_filter("Parsi on Iran", ["parsi"])
        True
    """
    title_lower = title.lower()
    for keyword in keywords:
        if keyword.lower() in title_lower:
            return True
    return False


def duration_is_acceptable(duration_text: str) -> bool:
    """
    Check whether a video duration meets the minimum threshold.

    Arguments:
        duration_text (str): Duration formatted as hh:mm:ss.

    Returns:
        bool: True when the duration is at least
            MINIMUM_TRANSCRIPT_DURATION_SECONDS.

    Example:
        >>> duration_is_acceptable("00:20:00")
        True
        >>> duration_is_acceptable("00:10:00")
        False
    """
    parts = duration_text.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    total_seconds = hours * 3600 + minutes * 60 + seconds
    is_acceptable = total_seconds >= MINIMUM_TRANSCRIPT_DURATION_SECONDS

    return is_acceptable


def format_time_window_label() -> str:
    """
    Build a compact local-time label for the current detection window.

    Arguments:
        None

    Returns:
        str: Human-readable window label, e.g.
            "2026-10-09 17:00 to 2026-10-10 17:00".

    Example:
        >>> label = format_time_window_label()
        >>> " to " in label
        True
    """
    now = datetime.now(LOCAL_TIMEZONE)
    end = now.replace(hour=START_HOUR, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=1)
    end = start + timedelta(days=1)
    start_label = start.strftime("%Y-%m-%d %H:%M")
    end_label = end.strftime("%Y-%m-%d %H:%M")
    label = f"{start_label} to {end_label}"

    return label


def print_detection_header() -> None:
    """
    Print the deterministic header of the stage 1 banner.

    Arguments:
        None

    Returns:
        None

    Example:
        >>> print_detection_header()
        ------------------------------------------------------------
        ################# Stage 1: Video Detection #################
        ------------------------------------------------------------
        Time window: 2026-10-09 17:00 to 2026-10-10 17:00
        Channel                       | Detected | Of Interest
        ------------------------------------------------------------
    """
    print("-" * 60)
    print("################# Stage 1: Video Detection #################")
    print("-" * 60)
    print(f"Time window: {format_time_window_label()}")
    print(f"{'Channel':<30} | {'Detected':>8} | {'Of Interest':>11}")
    print("-" * 60)


def print_detection_body(results: dict[str, dict[str, Any]]) -> None:
    """
    Print the per-channel counts and totals for stage 1.

    Arguments:
        results (dict[str, dict[str, Any]]): Results from detect_recent_videos().

    Returns:
        None

    Example:
        >>> print_detection_body({})
        ------------------------------------------------------------
        Total                          |        0 |           0
        ------------------------------------------------------------
    """
    total_detected = 0
    total_of_interest = 0

    for channel_id, data in results.items():
        channel_name = data["name"]
        videos = data["videos"]

        detected_count = len(videos)

        channel_filter = CHANNEL_TITLE_FILTERS.get(channel_id)
        of_interest_count = 0
        for video in videos:
            if channel_filter is not None:
                if title_matches_filter(video["title"], channel_filter) == False:
                    continue

            if duration_is_acceptable(video.get("duration", "00:00:00")) == False:
                continue

            of_interest_count += 1

        total_detected += detected_count
        total_of_interest += of_interest_count

        print(f"{channel_name:<30} | {detected_count:>8} | {of_interest_count:>11}")

    print("-" * 60)
    print(f"{'Total':<30} | {total_detected:>8} | {total_of_interest:>11}")
    print("-" * 60)


def format_elapsed_seconds(seconds: float) -> str:
    """
    Format an elapsed duration as seconds or minutes+seconds.

    Arguments:
        seconds (float): Elapsed time in seconds.

    Returns:
        str: Formatted duration, e.g. "5.2s" or "1m 5.2s".

    Example:
        >>> format_elapsed_seconds(5.2)
        '5.2s'
        >>> format_elapsed_seconds(65.2)
        '1m 5.2s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}m {remainder:.1f}s"


def build_detection_records(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Flatten detection results into downstream records.

    Arguments:
        results (dict[str, dict[str, Any]]): Results from detect_recent_videos().

    Returns:
        list[dict[str, Any]]: Flat detection records for downstream stages.

    Example:
        >>> records = build_detection_records({})
        >>> records
        []
    """
    records: list[dict[str, Any]] = []
    for channel_id, data in results.items():
        has_channel_error = bool(data["error"])
        # Preserve channel-level fetch errors by skipping that channel in flat output.
        if has_channel_error == True:
            logging.warning("Skipping channel %s: %s", data["name"], data["error"])
            continue

        for video in data["videos"]:
            channel_filter = CHANNEL_TITLE_FILTERS.get(channel_id)
            if channel_filter is not None:
                if title_matches_filter(video["title"], channel_filter) == False:
                    continue

            if duration_is_acceptable(video.get("duration", "00:00:00")) == False:
                continue

            records.append(
                {
                    "channel_id": channel_id,
                    "channel_name": data["name"],
                    "title": video["title"],
                    "duration": video.get("duration", "00:00:00"),
                    "url": video["url"],
                    "published_at": video["published_at"],
                    "description": video.get("description", ""),
                }
            )
    return records