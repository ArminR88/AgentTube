"""Stage runner for transcript summarization."""

from datetime import datetime
from pathlib import Path
import json
import logging
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.output_helper import build_output_directories, write_json  # noqa: E402
from helpers.transcript_summarization_helper import (
    build_summary_transcript_records,
    format_elapsed_seconds,
    print_summarization_body,
    print_summarization_header,
    summarize_transcript_records,
    write_summary_transcript_records,
)  # noqa: E402


def run_stage(
    summary_limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Summarize transcript records.

    Arguments:
        summary_limit (int | None): Optional limit on the number of summarized records.

    Returns:
        list[dict[str, Any]]: Records merged with summary results.

    Example:
        >>> run_stage(summary_limit=None)
        []
    """
    logging.info("[Stage 4] Transcript summarization started")
    print_summarization_header()
    start_time = time.time()

    output_dirs = build_output_directories()
    records_path = output_dirs["pipeline_summary"] / "02_transcript_detection.json"

    if not records_path.exists():
        raise SystemExit(f"Transcript detection input not found: {records_path}")

    with open(records_path, "r", encoding="utf-8") as file:
        records = json.load(file)

    summarized_records = summarize_transcript_records(
        records,
        str(output_dirs["transcripts"]),
        summary_limit=summary_limit,
    )

    summary_transcript_records = build_summary_transcript_records(summarized_records)
    write_summary_transcript_records(summary_transcript_records, output_dirs["transcript_summary"])

    # Compute success count explicitly to keep stage statistics easy to review.
    summary_success_count = 0
    for record in summarized_records:
        summary_result = record.get("summary_result", {})
        has_success = summary_result.get("success")
        if has_success == True:
            summary_success_count += 1

    summary_stats = {
        "stage": "transcript_summarization",
        "record_count": len(summarized_records),
        "summary_success_count": summary_success_count,
        "summary_limit": summary_limit,
        "generated_at": datetime.now().isoformat(),
    }
    write_json(output_dirs["pipeline_summary"] / "04_transcript_summarization.json", summary_stats)

    print_summarization_body(
        records,
        str(output_dirs["transcripts"]),
        str(output_dirs["transcript_summary"]),
    )

    elapsed_seconds = time.time() - start_time
    logging.info(
        "[Stage 4] Transcript summarization finished (%s)",
        format_elapsed_seconds(elapsed_seconds),
    )

    return summarized_records
