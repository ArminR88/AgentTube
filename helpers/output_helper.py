"""Helpers for organizing AgentTube pipeline outputs."""

from datetime import date, datetime
import json
from pathlib import Path
from typing import Any


def get_run_date() -> str:
    """
    Return the current run date in yyyy-mm-dd format.

    Arguments:
        None

    Returns:
        str: Current local date as a string.

    Example:
        >>> len(get_run_date())
        10
    """
    run_date = date.today().isoformat()

    result_run_date = run_date

    return result_run_date


def build_output_directories(root_name: str = "output_agenttube") -> dict[str, Path]:
    """
    Build the dated output directory structure for one pipeline run.

    Arguments:
        root_name (str): Top-level output folder name.

    Returns:
        dict[str, Path]: Named output directories for the current run.

    Example:
        >>> directories = build_output_directories()
        >>> "transcript_summary" in directories
        True
    """
    run_date = get_run_date()
    run_root = Path(root_name) / run_date

    directories = {
        "run_root": run_root,
        "transcripts": run_root / "transcripts",
        "transcript_summary": run_root / "transcript_summary",
        "fact_checking": run_root / "fact_checking",
        "pipeline_summary": run_root / "pipeline_summary",
        "verification": run_root / "verification",
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    result_directories = directories

    return result_directories


def write_json(path: Path, data: Any) -> Path:
    """
    Write JSON data to a file with stable formatting.

    Arguments:
        path (Path): Destination file path.
        data (Any): JSON-serializable data.

    Returns:
        Path: The path that was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)
        file.write("\n")

    return path


def write_text(path: Path, text: str) -> Path:
    """
    Write plain text to a file.

    Arguments:
        path (Path): Destination file path.
        text (str): Text content to write.

    Returns:
        Path: The path that was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)

    return path


def write_failure_artifacts(output_dirs: dict[str, Path], error_message: str) -> dict[str, Path]:
    """
    Write minimal output artifacts for a failed pipeline run.

    Arguments:
        output_dirs (dict[str, Path]): Output directories for the current run.
        error_message (str): Failure reason to record.

    Returns:
        dict[str, Path]: Paths of the artifacts that were written.
    """
    pipeline_summary = {
        "run_date": output_dirs["run_root"].name,
        "output_root": str(output_dirs["run_root"]),
        "stages": {
            "detection": {
                "stage": "detection",
                "status": "failed",
                "error": error_message,
            },
        },
    }
    verification_summary = {
        "passed": False,
        "generated_at": datetime.now().isoformat(),
        "messages": [error_message],
    }

    pipeline_summary_path = write_json(output_dirs["pipeline_summary"] / "pipeline_summary.json", pipeline_summary)
    verification_json_path = write_json(output_dirs["verification"] / "verification.json", verification_summary)
    verification_log_path = write_text(output_dirs["verification"] / "verification.log", f"Passed: no\n{error_message}\n")

    artifact_paths = {
        "pipeline_summary": pipeline_summary_path,
        "verification_json": verification_json_path,
        "verification_log": verification_log_path,
    }

    return artifact_paths