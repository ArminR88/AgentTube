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
        >>> stats = run_stage(inter_video_delay=None)
        >>> isinstance(stats, dict)
        True
    """
    output_dirs = build_output_directories()
    records_path = output_dirs["pipeline_summary"] / "02_transcript_detection.json"

    with open(records_path, "r", encoding="utf-8") as file:
        records = json.load(file)

    download_stats = download_transcripts_from_records(
        records,
        str(output_dirs["transcripts"]),
        inter_video_delay=inter_video_delay,
    )
    write_json(output_dirs["pipeline_summary"] / "03_download_summary.json", download_stats)

    return download_stats