"""Stage runner for detecting transcript availability on videos.

This stage receives the video-detection records and enriches them with
transcript metadata for downstream processing.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.transcript_detection_helper import detect_transcripts_for_records  # noqa: E402
from helpers.shared_helper import setup_logging  # noqa: E402


def run_stage(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """
    Run the transcript detection stage for the provided records.

    Arguments:
        records (list[dict[str, object]]): Flat records from the video detection stage.

    Returns:
        list[dict[str, object]]: The same records with transcript metadata added.

    Example:
        >>> run_stage([])
        []
    """
    enriched_records = detect_transcripts_for_records(records)

    return enriched_records


def main() -> None:
    """
    CLI entry point for the transcript detection stage.

    Arguments:
        None

    Returns:
        None

    Example:
        $ python stage/transcript_detection_stage.py --json
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run the transcript detection stage")
    parser.add_argument("--json", action="store_true", help="Print records as JSON")
    parser.add_argument("records_file", nargs="?", help="JSON file with detection records")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    if not args.records_file:
        raise SystemExit("Provide a JSON file with detection records.")

    with open(args.records_file, "r", encoding="utf-8") as file:
        records = json.load(file)

    enriched_records = run_stage(records)

    if args.json:
        print(json.dumps(enriched_records, indent=2, default=str))
        return

    for index, record in enumerate(enriched_records, 1):
        available = "yes" if record.get("transcript_available") else "no"
        print(f"{index}. {record['channel_name']} | {record['title']} | transcript: {available}")


if __name__ == "__main__":
    main()