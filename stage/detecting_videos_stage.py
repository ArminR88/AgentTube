"""Stage runner for detecting recent YouTube videos."""

import io
import logging
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.detecting_videos_helper import (  # noqa: E402
    build_detection_records,
    detect_recent_videos,
    format_elapsed_seconds,
    get_api_key,
    print_detection_body,
    print_detection_header,
)
from helpers.output_helper import (  # noqa: E402
    build_monitoring_banners_directory,
    build_output_directories,
    write_json,
    write_text,
)


def run_stage() -> list[dict[str, object]]:
    """
    Run the detection stage and return flat schema records.

    Arguments:
        None

    Returns:
        list[dict[str, object]]: Flat detection records with these fields:
            channel_id, channel_name, title, duration, url, published_at, description.

    Example:
        >>> records = run_stage()
        >>> records
        [
            {
                'channel_id': 'UCDkEYb-TXJVWLvOokshtlsw',
                'channel_name': 'Judge_Napolitano',
                'title': 'Prof. Jeffrey Sachs  :  Can Dr. Fauci Tell The Truth?',
                'url': 'https://www.youtube.com/watch?v=NdHZDJmaYqg',
                'published_at': '2026-07-30T10:28:36Z',
                'description': 'Prof. Jeffrey Sachs : Can Dr. Fauci Tell The Truth?'
            }
        ]
    """
    logging.info("[Stage 1] Video detection started")
    start_time = time.time()

    api_key = get_api_key()
    if not api_key:
        raise SystemExit("YOUTUBE_API_KEY is not set.")

    results = detect_recent_videos(api_key)

    banner_buffer = io.StringIO()
    print_detection_header(output_stream=banner_buffer)
    print_detection_body(results, output_stream=banner_buffer)
    banner_text = banner_buffer.getvalue()

    print(banner_text)
    banners_dir = build_monitoring_banners_directory()
    write_text(banners_dir / "stage_1.txt", banner_text)

    output_dirs = build_output_directories()

    all_videos = []
    for channel_id, data in results.items():
        for video in data["videos"]:
            all_videos.append({
                "channel_id": channel_id,
                "channel_name": data["name"],
                "title": video["title"],
                "duration": video.get("duration", "00:00:00"),
                "url": video["url"],
                "published_at": video["published_at"],
                "description": video.get("description", ""),
            })

    write_json(output_dirs["pipeline_summary"] / "01_detection_all.json", all_videos)

    records = build_detection_records(results)

    write_json(output_dirs["pipeline_summary"] / "01_detection.json", records)

    elapsed_seconds = time.time() - start_time
    logging.info(
        "[Stage 1] Video detection finished (%s)",
        format_elapsed_seconds(elapsed_seconds),
    )

    return records