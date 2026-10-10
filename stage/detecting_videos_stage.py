"""Stage runner for detecting recent YouTube videos."""

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
from helpers.output_helper import build_output_directories, write_json  # noqa: E402


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
    print_detection_header()
    start_time = time.time()

    api_key = get_api_key()
    if not api_key:
        raise SystemExit("YOUTUBE_API_KEY is not set.")

    results = detect_recent_videos(api_key)
    print_detection_body(results)
    records = build_detection_records(results)

    output_dirs = build_output_directories()
    write_json(output_dirs["pipeline_summary"] / "01_detection.json", records)

    elapsed_seconds = time.time() - start_time
    logging.info(
        "[Stage 1] Video detection finished (%s)",
        format_elapsed_seconds(elapsed_seconds),
    )

    return records