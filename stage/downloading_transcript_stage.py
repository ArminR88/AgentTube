"""Stage runner for downloading transcripts from detected videos.

This stage loads transcript-detection records and downloads transcripts
for each video using the shared downloader helper.
"""

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.downloading_transcript_helper import download_transcripts_from_records  # noqa: E402
from helpers.output_helper import build_output_directories, write_json  # noqa: E402
from helpers.shared_helper import setup_logging  # noqa: E402


def run_stage(
    inter_video_delay: int | None = None,
) -> dict[str, int]:
    """
    Run the download stage using default pipeline locations.

    Arguments:
        inter_video_delay (int | None): Seconds to sleep between transcript downloads.
            Uses downloader default when not provided.

    Returns:
        dict[str, int]: Download statistics.

    Example:
        >>> run_stage([{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}], output_dir="transcripts")
        {'success': 1, 'failed': 0, 'total': 1}
    """
    output_dirs = build_output_directories()
    records_path = output_dirs["pipeline_summary"] / "02_transcript_detection.json"

    if not records_path.exists():
        raise SystemExit(f"Transcript detection input not found: {records_path}")

    with open(records_path, "r", encoding="utf-8") as file:
        records = json.load(file)

    download_stats = download_transcripts_from_records(
        records,
        str(output_dirs["transcripts"]),
        inter_video_delay=inter_video_delay,
    )
    write_json(output_dirs["pipeline_summary"] / "03_download_summary.json", download_stats)

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

    parser = argparse.ArgumentParser(description="Run the transcript download stage")
    parser.add_argument("--json", action="store_true", help="Print the download stats as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--inter-video-delay",
        type=int,
        default=None,
        help="Seconds to sleep between transcript downloads",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    stats = run_stage(inter_video_delay=args.inter_video_delay)

    if args.json:
        print(json.dumps(stats, indent=2, default=str))
        return

    print(f"Download summary: {stats['success']} successful, {stats['failed']} failed, {stats['total']} total")


if __name__ == "__main__":
    main()