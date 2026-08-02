"""Stage runner for transcript summarization."""

from pathlib import Path
import json
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.shared_helper import setup_logging  # noqa: E402
from helpers.transcript_summarization_helper import (
    build_summary_transcript_records,
    summarize_transcript_records,
    write_summary_transcript_records,
)  # noqa: E402


def run_stage(
    records: list[dict[str, Any]],
    transcripts_dir: str = "transcripts",
    api_key: str | None = None,
    summary_limit: int | None = None,
    summary_output_dir: str | None = None,
) -> list[dict[str, Any]]:
    """
    Summarize transcript records.

    Arguments:
        records (list[dict[str, Any]]): Transcript detection records.
        transcripts_dir (str): Directory containing transcript files.
        api_key (str | None): DeepSeek API key.
        summary_limit (int | None): Optional limit on the number of summarized records.
        summary_output_dir (str | None): Optional directory for per-transcript summary files.

    Returns:
        list[dict[str, Any]]: Records merged with summary results.

    Example:
        >>> run_stage([])
        []
    """
    summarized_records = summarize_transcript_records(
        records,
        transcripts_dir,
        api_key=api_key,
        summary_limit=summary_limit,
    )

    if summary_output_dir is not None:
        summary_transcript_records = build_summary_transcript_records(summarized_records)
        write_summary_transcript_records(summary_transcript_records, summary_output_dir)

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


def main() -> None:
    """
    Run the CLI entry point.

    Arguments:
        None

    Returns:
        None

    Example:
        $ python stage/transcript_summarization_stage.py records.json --transcripts-dir output_agenttube/2026-07-31/transcripts
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run the transcript summarization stage")
    parser.add_argument("records_file", help="Path to a JSON file with transcript records")
    parser.add_argument("--transcripts-dir", default="transcripts", help="Directory containing transcript files")
    parser.add_argument("--summary-limit", type=int, help="Optional limit on the number of records to summarize")
    parser.add_argument(
        "--summary-output-dir",
        default="transcript_summary",
        help="Directory for per-transcript summary files",
    )
    parser.add_argument("--json", action="store_true", help="Print the summarized records as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    records = _load_records(args.records_file)
    summarized_records = run_stage(
        records,
        transcripts_dir=args.transcripts_dir,
        summary_limit=args.summary_limit,
        summary_output_dir=args.summary_output_dir,
    )

    if args.json:
        summary_transcript_records = build_summary_transcript_records(summarized_records)
        print(json.dumps(summary_transcript_records, indent=2, default=str))
        return

    for index, record in enumerate(summarized_records, 1):
        summary_result = record.get("summary_result", {})
        status = "yes" if summary_result.get("success") else "no"
        print(f"{index}. {record['channel_name']} | {record['title']} | summary: {status}")


if __name__ == "__main__":
    main()