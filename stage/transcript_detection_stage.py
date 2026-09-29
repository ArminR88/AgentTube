"""Stage runner for detecting transcript availability on videos.

This stage loads video-detection records from disk and enriches them with
transcript metadata for downstream processing.
"""

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.transcript_detection_helper import detect_transcripts_for_records  # noqa: E402
from helpers.output_helper import build_output_directories, write_json  # noqa: E402
from helpers.shared_helper import setup_logging  # noqa: E402


def _load_records(path: Path) -> list[dict[str, object]]:
    """
    Load records from JSON.

    Arguments:
        path (Path): Path to JSON input.

    Returns:
        list[dict[str, object]]: Loaded records.
    """
    with open(path, "r", encoding="utf-8") as file:
        records = json.load(file)

    return records


def run_stage() -> list[dict[str, object]]:
    """
    Run the transcript detection stage using default pipeline locations.

    Arguments:
        None

    Returns:
        list[dict[str, object]]: The same records with transcript metadata added.

    Example:
        >>> run_stage([])
        []
    """
    output_dirs = build_output_directories()
    records_path = output_dirs["pipeline_summary"] / "01_detection.json"

    if not records_path.exists():
        raise SystemExit(f"Detection input not found: {records_path}")

    records = _load_records(records_path)
    enriched_records = detect_transcripts_for_records(records)
    write_json(output_dirs["pipeline_summary"] / "02_transcript_detection.json", enriched_records)

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

    parser = argparse.ArgumentParser(description="Run the transcript detection stage")
    parser.add_argument("--json", action="store_true", help="Print records as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    enriched_records = run_stage()

    if args.json:
        print(json.dumps(enriched_records, indent=2, default=str))
        return

    for index, record in enumerate(enriched_records, 1):
        available = "yes" if record.get("transcript_available") else "no"
        print(f"{index}. {record['channel_name']} | {record['title']} | transcript: {available}")


if __name__ == "__main__":
    main()