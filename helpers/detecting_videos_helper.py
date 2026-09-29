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
VIDEO_DURATION_PATTERN = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)

CHANNEL_IDS = [
    "UCOxLhz6B_elvLflntSEfnzA",  # Danny_Haiphong
    "UCDkEYb-TXJVWLvOokshtlsw",  # Judge_Napolitano
    "UCZFCDIHTe9HGxtIuVDpBz7g",  # Glenn Diesen
    "UCWDN5zr5ttctoIAhZwW6tcQ",  # Daniel Davis / Deep Dive
]

CHANNEL_NAMES = {
    "UCOxLhz6B_elvLflntSEfnzA": "Danny_Haiphong",
    "UCDkEYb-TXJVWLvOokshtlsw": "Judge_Napolitano",
    "UCZFCDIHTe9HGxtIuVDpBz7g": "Glenn_Diesen",
    "UCWDN5zr5ttctoIAhZwW6tcQ": "Daniel_Davis",
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
            logging.info(
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

    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
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
        logging.info("Checking channel: %s", channel_name)
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