"""Stage runner for transcript summarization."""

from datetime import datetime
from pathlib import Path
import json
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.output_helper import build_output_directories, write_json  # noqa: E402
from helpers.transcript_summarization_helper import (
    build_summary_transcript_records,
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
        >>> run_stage([])
        []
    """
    output_dirs = build_output_directories()
    records_path = output_dirs["pipeline_summary"] / "02_transcript_detection.json"

    if not records_path.exists():
        raise SystemExit(f"Transcript detection input not found: {records_path}")

    records = _load_records(str(records_path))
    summarized_records = summarize_transcript_records(
        records,
        str(output_dirs["transcripts"]),
        summary_limit=summary_limit,
    )

    summary_transcript_records = build_summary_transcript_records(summarized_records)
    write_summary_transcript_records(summary_transcript_records, output_dirs["transcript_summary"])

    summary_stats = {
        "stage": "transcript_summarization",
        "record_count": len(summarized_records),
        "summary_success_count": sum(
            1 for record in summarized_records if record.get("summary_result", {}).get("success")
        ),
        "summary_limit": summary_limit,
        "generated_at": datetime.now().isoformat(),
    }
    write_json(output_dirs["pipeline_summary"] / "04_transcript_summarization.json", summary_stats)

    return summarized_records


def _load_records(records_file: str) -> list[dict[str, Any]]:
    """
    Load transcript records from JSON.

    Arguments:
        records_file (str): Path to a JSON file with transcript records.

    Returns:
        list[dict[str, Any]]: Transcript detection records.

    Example:
        >>> isinstance(_load_records, object)
        True
    """
    with open(records_file, "r", encoding="utf-8") as file:
        records = json.load(file)

    return records

