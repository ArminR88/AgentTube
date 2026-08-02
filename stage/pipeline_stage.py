"""Stage runner for the full AgentTube pipeline."""

from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.output_helper import build_output_directories, write_failure_artifacts, write_json, write_text  # noqa: E402
from helpers.shared_helper import setup_logging  # noqa: E402
from helpers.transcript_summarization_helper import build_summary_transcript_records  # noqa: E402
from helpers.transcript_fact_checking_helper import build_fact_check_stats  # noqa: E402
from stage.detecting_videos_stage import run_stage as run_detection_stage  # noqa: E402
from stage.downloading_transcript_stage import run_stage as run_download_stage  # noqa: E402
from stage.transcript_fact_checking_stage import run_stage as run_fact_check_stage  # noqa: E402
from stage.transcript_detection_stage import run_stage as run_transcript_detection_stage  # noqa: E402
from stage.transcript_summarization_stage import run_stage as run_summary_stage  # noqa: E402


def run_stage(
    verbose: bool = False,
    run_detection: bool = True,
    run_transcript_detection: bool = True,
    run_download: bool = True,
    run_summary: bool = True,
    run_fact_check: bool = True,
    search_api_key: str | None = None,
    search_provider: str | None = None,
    summary_limit: int | None = None,
    output_root: str = "output_agenttube",
) -> dict[str, object]:
    """
    Run the full AgentTube pipeline and persist dated output artifacts.

    Arguments:
        verbose (bool): Enable debug-level logging if True. Default: False.
        run_detection (bool): Run the detection stage if True.
        run_transcript_detection (bool): Run the transcript detection stage if True.
        run_download (bool): Run transcript download if True.
        run_summary (bool): Run transcript summarization if True.
        run_fact_check (bool): Run transcript fact checking if True.
        search_api_key (str | None): Optional API key that enables Responses API web search.
        search_provider (str | None): Reserved compatibility argument for future search workflows.
        summary_limit (int | None): Optional cap on summarized records.
        output_root (str): Top-level output folder name.

    Returns:
        dict[str, object]: Pipeline payload containing records, outputs, and stats.

    Example:
        >>> payload = run_stage()
        >>> "output_root" in payload
        True
    """
    setup_logging(verbose)
    output_dirs = build_output_directories(output_root)

    try:
        records = run_detection_stage() if run_detection else []
    except SystemExit as exc:
        write_failure_artifacts(output_dirs, str(exc))
        raise

    if run_transcript_detection and not run_detection:
        raise SystemExit("Transcript detection requires detection to run first.")

    if run_transcript_detection:
        transcript_records = run_transcript_detection_stage(records)
    else:
        transcript_records = records
    download_stats = None

    if run_download and not run_transcript_detection:
        raise SystemExit("Download requires transcript detection to run first.")

    if run_download:
        download_stats = run_download_stage(transcript_records, output_dir=str(output_dirs["transcripts"]))

    if run_summary and not run_download:
        raise SystemExit("Summarization requires transcript download to run first.")

    if run_fact_check and not run_summary:
        raise SystemExit("Fact checking requires transcript summarization to run first.")

    summary_records = []
    summary_transcript_records = []
    summary_stats = None
    fact_check_records = []
    fact_check_stats = None

    if run_summary:
        summary_records = run_summary_stage(
            transcript_records,
            transcripts_dir=str(output_dirs["transcripts"]),
            summary_limit=summary_limit,
            summary_output_dir=str(output_dirs["transcript_summary"]),
        )
        summary_transcript_records = build_summary_transcript_records(summary_records)
        expected_summary_count = len(transcript_records)
        if summary_limit is not None:
            expected_summary_count = min(expected_summary_count, summary_limit)
        summary_stats = {
            "stage": "transcript_summarization",
            "record_count": len(summary_records),
            "summary_success_count": sum(1 for record in summary_records if record.get("summary_result", {}).get("success")),
            "summary_limit": summary_limit,
            "generated_at": datetime.now().isoformat(),
        }

    if run_fact_check:
        fact_check_records = run_fact_check_stage(
            summary_transcript_records,
            output_dir=str(output_dirs["fact_checking"]),
            search_api_key=search_api_key,
            search_provider=search_provider,
        )
        fact_check_stats = build_fact_check_stats(summary_transcript_records, fact_check_records)

    detection_summary = {
        "stage": "detection",
        "record_count": len(records),
        "generated_at": datetime.now().isoformat(),
    }
    transcript_summary = {
        "stage": "transcript_detection",
        "input_record_count": len(records),
        "record_count": len(transcript_records),
        "transcript_available_count": sum(1 for record in transcript_records if record.get("transcript_available")),
        "generated_at": datetime.now().isoformat(),
    }
    download_summary = None
    if download_stats is not None:
        download_summary = {
            "stage": "download",
            **download_stats,
            "generated_at": datetime.now().isoformat(),
        }

    pipeline_summary = {
        "run_date": output_dirs["run_root"].name,
        "output_root": str(output_dirs["run_root"]),
        "stages": {
            "detection": detection_summary,
            "transcript_detection": transcript_summary,
            "download": download_summary,
            "transcript_summarization": summary_stats,
            "fact_checking": fact_check_stats,
        },
    }

    verification_passed = True
    verification_messages = []
    if len(records) != len(transcript_records):
        verification_messages.append(
            f"Filtered final results: detection={len(records)} transcript_detection={len(transcript_records)}"
        )
    if download_stats is not None and download_stats["success"] + download_stats["failed"] != download_stats["total"]:
        verification_passed = False
        verification_messages.append("Download stats do not add up to the reported total.")
    expected_summary_count = len(transcript_records)
    if summary_limit is not None:
        expected_summary_count = min(expected_summary_count, summary_limit)
    if run_summary and len(summary_records) != expected_summary_count:
        verification_passed = False
        verification_messages.append(
            f"Summary record mismatch: expected={expected_summary_count} summarization={len(summary_records)}"
        )
    if run_fact_check and len(fact_check_records) != len(summary_transcript_records):
        verification_passed = False
        verification_messages.append(
            f"Fact-check record mismatch: expected={len(summary_transcript_records)} fact_checking={len(fact_check_records)}"
        )

    verification_summary = {
        "passed": verification_passed,
        "generated_at": datetime.now().isoformat(),
        "messages": verification_messages,
    }

    write_json(output_dirs["pipeline_summary"] / "01_detection.json", records)
    write_json(output_dirs["pipeline_summary"] / "02_transcript_detection.json", transcript_records)
    if download_summary is not None:
        write_json(output_dirs["pipeline_summary"] / "03_download_summary.json", download_summary)
    if summary_stats is not None:
        write_json(output_dirs["pipeline_summary"] / "04_transcript_summarization.json", summary_stats)
    if fact_check_stats is not None:
        write_json(output_dirs["pipeline_summary"] / "05_fact_checking.json", fact_check_stats)
    write_json(output_dirs["pipeline_summary"] / "pipeline_summary.json", pipeline_summary)
    write_json(output_dirs["verification"] / "verification.json", verification_summary)
    write_text(
        output_dirs["verification"] / "verification.log",
        "Passed: " + ("yes" if verification_passed else "no") + "\n"
        + "\n".join(verification_messages or ["All checks passed."])
        + "\n",
    )

    pipeline_payload = {
        "records": records,
        "transcript_records": transcript_records,
        "summary_records": summary_records,
        "summary_transcript_records": summary_transcript_records,
        "fact_check_records": fact_check_records,
        "fact_check_stats": fact_check_stats,
        "download_stats": download_stats,
        "output_root": str(output_dirs["run_root"]),
    }

    return pipeline_payload