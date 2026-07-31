"""Stage runner for transcript summarization."""

from pathlib import Path
import json
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.shared_helper import setup_logging  # noqa: E402
from helpers.transcript_summarization_helper import TranscriptSummaryResult, summarize_transcript  # noqa: E402


def run_stage(transcript: str | None, metadata: dict[str, Any] | None = None) -> TranscriptSummaryResult:
    """
    Summarize one transcript.

    Arguments:
        transcript (str | None): Transcript text to summarize.
        metadata (dict[str, Any] | None): Optional metadata dictionary.

    Returns:
        TranscriptSummaryResult: Structured summarization result.

    Example:
        >>> run_stage("A short transcript.")
        TranscriptSummaryResult(...)
    """
    summary_result = summarize_transcript(transcript, metadata=metadata)

    return summary_result


def _load_metadata(metadata_file: str | None) -> dict[str, Any] | None:
    """
    Load optional metadata.

    Arguments:
        metadata_file (str | None): Path to a JSON metadata file.

    Returns:
        dict[str, Any] | None: Loaded metadata dictionary or None.

    Example:
        >>> _load_metadata(None) is None
        True
    """
    if not metadata_file:
        metadata = None

        return metadata

    with open(metadata_file, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return metadata


def main() -> None:
    """
    Run the CLI entry point.

    Arguments:
        None

    Returns:
        None

    Example:
        $ python stage/transcript_summarization_stage.py transcript.txt --json
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run the transcript summarization stage")
    parser.add_argument("transcript_file", help="Path to a transcript text file")
    parser.add_argument("--metadata-file", help="Optional JSON file with transcript metadata")
    parser.add_argument("--json", action="store_true", help="Print the summary result as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    with open(args.transcript_file, "r", encoding="utf-8") as file:
        transcript = file.read()

    metadata = _load_metadata(args.metadata_file)
    summary_result = run_stage(transcript, metadata=metadata)

    if args.json:
        print(summary_result.model_dump_json(indent=2))
        return

    if not summary_result.success:
        print(f"Summary failed: {summary_result.error}")
        return

    print(summary_result.summary)


if __name__ == "__main__":
    main()