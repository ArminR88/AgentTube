"""Stage runner for downloading transcripts from detected videos.

This stage receives the detection records and downloads transcripts
for each video using the shared downloader helper.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.downloading_transcript_helper import download_transcripts_from_records  # noqa: E402
from helpers.shared_helper import setup_logging  # noqa: E402


def run_stage(records: list[dict[str, object]], output_dir: str = "transcripts") -> dict[str, int]:
    """
    Run the download stage for the provided records.

    Arguments:
        records (list[dict[str, object]]): Detection records from the previous stage.
        output_dir (str): Directory to save transcript files.

    Returns:
        dict[str, int]: Download statistics.

    Example:
        >>> run_stage([{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}], output_dir="transcripts")
        {'success': 1, 'failed': 0, 'total': 1}
    """
    download_stats = download_transcripts_from_records(records, output_dir)

    return download_stats


def main() -> None:
    """
    CLI entry point for the download stage.

    Arguments:
        None

    Returns:
        None

    Example:
        $ python stage/downloading_transcript_stage.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
    """
    import argparse
    import json
    import logging

    parser = argparse.ArgumentParser(description="Run the transcript download stage")
    parser.add_argument("--json", action="store_true", help="Print the download stats as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--output-dir", default="transcripts", help="Directory to save transcript files")
    parser.add_argument("urls", nargs="*", help="Video URLs to download")
    args = parser.parse_args()

    setup_logging(args.verbose)

    records = [{"url": url} for url in args.urls]
    stats = run_stage(records, output_dir=args.output_dir)

    if args.json:
        print(json.dumps(stats, indent=2, default=str))
        return

    print(f"Download summary: {stats['success']} successful, {stats['failed']} failed, {stats['total']} total")


if __name__ == "__main__":
    main()