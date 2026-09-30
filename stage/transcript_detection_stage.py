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


def _load_records(path: Path) -> list[dict[str, object]]:
    """
    Load records from JSON.

    Arguments:
        path (Path): Path to JSON input.

    Returns:
        list[dict[str, object]]: Loaded records.

    Example:
        >>> isinstance(_load_records, object)
        True
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
        >>> records = run_stage()
        >>> isinstance(records, list)
        True
    """
    output_dirs = build_output_directories()
    records_path = output_dirs["pipeline_summary"] / "01_detection.json"

    if not records_path.exists():
        raise SystemExit(f"Detection input not found: {records_path}")

    records = _load_records(records_path)
    enriched_records = detect_transcripts_for_records(records)
    write_json(output_dirs["pipeline_summary"] / "02_transcript_detection.json", enriched_records)

    return enriched_records