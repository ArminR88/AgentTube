"""Stage runner for detecting transcript availability on videos.

This stage loads video-detection records from disk and enriches them with
transcript metadata for downstream processing.
"""

import json
import logging
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.transcript_detection_helper import (  # noqa: E402
    detect_transcripts_for_records,
    format_elapsed_seconds,
    print_transcript_detection_body,
    print_transcript_detection_header,
)
from helpers.output_helper import build_output_directories, write_json  # noqa: E402


def run_stage() -> list[dict[str, object]]:
    """
    Run the transcript detection stage using default pipeline locations.

    Arguments:
        None

    Returns:
        list[dict[str, object]]: The same records with transcript metadata added.

    Example:
        >>> records = run_stage()
        >>> isinstance(records, list)
        True
    """
    logging.info("[Stage 2] Transcript detection started")
    print_transcript_detection_header()
    start_time = time.time()

    output_dirs = build_output_directories()
    records_path = output_dirs["pipeline_summary"] / "01_detection.json"

    with open(records_path, "r", encoding="utf-8") as file:
        records = json.load(file)

    enriched_records = detect_transcripts_for_records(records)

    write_json(output_dirs["pipeline_summary"] / "02_transcript_detection.json", enriched_records)

    print_transcript_detection_body(enriched_records)

    elapsed_seconds = time.time() - start_time
    logging.info(
        "[Stage 2] Transcript detection finished (%s)",
        format_elapsed_seconds(elapsed_seconds),
    )

    return enriched_records