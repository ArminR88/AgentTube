"""Stage runner for detecting recent YouTube videos.

This stage is the executable boundary for the detection step.
It reads the API key, runs the detector, and returns flat records.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.detecting_videos_helper import (  # noqa: E402
    build_detection_records,
    detect_recent_videos,
    get_api_key,
)
from helpers.shared_helper import setup_logging  # noqa: E402


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
    api_key = get_api_key()
    if not api_key:
        raise SystemExit("YOUTUBE_API_KEY is not set.")

    results = detect_recent_videos(api_key)
    records = build_detection_records(results)

    return records


def main() -> None:
    """
    CLI entry point for the detection stage.

    Arguments:
        None

    Returns:
        None

    Example:
        $ python stage/detecting_videos_stage.py --json
    """
    import argparse
    import json
    import logging

    parser = argparse.ArgumentParser(description="Run the detection stage")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--json", action="store_true", help="Print records as JSON")
    args = parser.parse_args()

    setup_logging(args.verbose)

    try:
        records = run_stage()
    except SystemExit as exc:
        logging.error(str(exc))
        raise

    if args.json:
        print(json.dumps(records, indent=2, default=str))
        return

    for index, record in enumerate(records, 1):
        print(f"{index}. {record['channel_name']} | {record['title']} | duration: {record['duration']}")


if __name__ == "__main__":
    main()