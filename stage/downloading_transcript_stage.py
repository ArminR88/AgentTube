"""Stage runner for downloading transcripts from detected videos.

This stage loads transcript-detection records and downloads transcripts
for each video using the shared downloader helper.
"""

import json
import logging
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.downloading_transcript_helper import (  # noqa: E402
    download_transcripts_from_records,
    format_elapsed_seconds,
    print_transcript_download_body,
    print_transcript_download_header,
)
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
    logging.info("[Stage 3] Transcript download started")
    print_transcript_download_header()
    start_time = time.time()

    output_dirs = build_output_directories()
    records_path = output_dirs["pipeline_summary"] / "02_transcript_detection.json"

    with open(records_path, "r", encoding="utf-8") as file:
        records = json.load(file)

    download_stats = download_transcripts_from_records(
        records,
        str(output_dirs["transcripts"]),
        inter_video_delay=inter_video_delay,
    )
    download_total = int(download_stats.get("total", 0))
    download_failed = int(download_stats.get("failed", 0))
    download_success = int(download_stats.get("success", 0))
    download_processed = download_success + download_failed
    download_unprocessed = download_total - download_processed
    if download_unprocessed < 0:
        download_unprocessed = 0

    download_incomplete = download_failed > 0 or download_success == 0 or download_unprocessed > 0
    if download_incomplete == True:
        download_stats["status"] = "incomplete"
    else:
        download_stats["status"] = "complete"
    download_stats["processed"] = download_processed
    download_stats["unprocessed"] = download_unprocessed
    write_json(output_dirs["pipeline_summary"] / "03_download_summary.json", download_stats)
    write_json(output_dirs["pipeline_summary"] / "03_download_summary.json", download_stats)

    print_transcript_download_body(records, str(output_dirs["transcripts"]))

    elapsed_seconds = time.time() - start_time
    logging.info(
        "[Stage 3] Transcript download finished (%s)",
        format_elapsed_seconds(elapsed_seconds),
    )

    return download_stats