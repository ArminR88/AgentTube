"""Stage runner for the full AgentTube pipeline."""

from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.output_helper import build_output_directories, build_topics_output_directories, write_failure_artifacts, write_json, write_text
from helpers.shared_helper import setup_logging
from helpers.transcript_summarization_helper import build_summary_transcript_records
from helpers.summary_of_summaries_helper import build_topics_stats, extract_topics, load_summary_files, write_topics
from helpers.news_script_generator_helper import build_digest_stats, generate_perspective_digest, write_perspective_digest
from stage.detecting_videos_stage import run_stage as run_detection_stage
from stage.downloading_transcript_stage import run_stage as run_download_stage
from stage.transcript_detection_stage import run_stage as run_transcript_detection_stage
from stage.transcript_summarization_stage import run_stage as run_summary_stage


def run_stage(
    verbose: bool = False,
    run_detection: bool = True,
    run_transcript_detection: bool = True,
    run_download: bool = True,
    run_summary: bool = True,
    run_topics: bool = True,
    run_script: bool = True,
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
        run_topics (bool): Run topic extraction if True.
        run_script (bool): Run perspective digest generation if True.
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
    claims_dirs = build_topics_output_directories(output_root)

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

    if run_topics and not run_summary:
        raise SystemExit("Topic extraction requires transcript summarization to run first.")

    if run_script and not run_topics:
        raise SystemExit("Digest generation requires topic extraction to run first.")

    summary_records = []
    summary_transcript_records = []
    summary_stats = None
    topics_result = None
    topics_stats = None
    digest_result = None
    digest_stats = None

    # Stage 4: Transcript Summarization
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

    # Stage 5: Topic Extraction
    if run_topics:
        summary_files = load_summary_files(str(output_dirs["transcript_summary"]))

        if summary_files:
            import os

            api_key = os.environ.get("DEEPSEEK_API_KEY")
            topics_draft, tokens_used, cost = extract_topics(summary_files, api_key=api_key)
            topics_path = write_topics(topics_draft, str(claims_dirs["topics"]), tokens_used, cost)
            topics_stats = build_topics_stats(len(summary_files), topics_draft)

            topics_payload = [
                topic.model_dump() if hasattr(topic, "model_dump") else topic.dict()
                for topic in topics_draft.topics
            ]
            perspective_count = sum(len(topic.perspectives) for topic in topics_draft.topics)

            topics_result = {
                "topics": topics_payload,
                "topic_count": len(topics_draft.topics),
                "perspective_count": perspective_count,
                "tokens_used": tokens_used,
                "cost": cost,
                "output_path": str(topics_path),
            }
        else:
            topics_stats = {
                "stage": "topic_extraction",
                "summary_record_count": 0,
                "topic_count": 0,
                "perspective_count": 0,
                "theme_count": 0,
                "error": "No summary files found",
                "generated_at": datetime.now().isoformat(),
            }
            topics_result = {
                "topics": [],
                "topic_count": 0,
                "perspective_count": 0,
                "tokens_used": 0,
                "cost": 0.0,
                "output_path": "",
            }

    # Stage 6: Perspective Digest Generator
    if run_script and topics_result and topics_result.get("topic_count", 0) > 0:
        import os

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        topics_data = {
            "topics": topics_result["topics"],
            "metadata": {
                "topic_count": topics_result["topic_count"],
                "perspective_count": topics_result["perspective_count"],
                "tokens_used": topics_result["tokens_used"],
                "cost": topics_result["cost"],
            },
        }

        digest_text, tokens_used, cost = generate_perspective_digest(
            topics_data,
            api_key=api_key,
            temperature=0.7,
            max_tokens=8000,
        )

        text_path, json_path = write_perspective_digest(
            digest_text,
            str(claims_dirs["news_script"]),
            topics_data,
            tokens_used,
            cost,
        )

        word_count = len(digest_text.split())
        topic_count = int(topics_result.get("topic_count", 0))
        perspective_count = int(topics_result.get("perspective_count", 0))
        digest_stats = build_digest_stats(topic_count, perspective_count, word_count, tokens_used, cost)

        digest_result = {
            "digest": digest_text,
            "word_count": word_count,
            "tokens_used": tokens_used,
            "cost": cost,
            "text_output_path": str(text_path),
            "json_output_path": str(json_path),
        }
    elif run_script:
        digest_stats = {
            "stage": "perspective_digest_generation",
            "topic_count": 0,
            "perspective_count": 0,
            "word_count": 0,
            "tokens_used": 0,
            "cost": 0.0,
            "error": "No topics available",
            "generated_at": datetime.now().isoformat(),
        }

    # Build pipeline summary
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
            "topic_extraction": topics_stats,
            "perspective_digest_generation": digest_stats,
        },
    }

    # Verification
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
    if run_topics:
        topic_count = topics_result.get("topic_count", 0) if topics_result else 0
        if topic_count < 3:
            verification_passed = False
            verification_messages.append(f"Expected at least 3 topics, got {topic_count}")
        elif topic_count < 6:
            verification_messages.append(f"Only {topic_count} topics extracted")
    if run_script:
        digest_words = digest_result.get("word_count", 0) if digest_result else 0
        if digest_words < 100:
            verification_passed = False
            verification_messages.append(f"Digest too short: {digest_words} words")
        elif digest_words < 500:
            verification_messages.append(f"Digest only {digest_words} words")
    if run_script and not digest_result:
        verification_passed = False
        verification_messages.append("Perspective digest generation did not produce an output payload.")

    verification_summary = {
        "passed": verification_passed,
        "generated_at": datetime.now().isoformat(),
        "messages": verification_messages,
    }

    # Write all output files
    write_json(output_dirs["pipeline_summary"] / "01_detection.json", records)
    write_json(output_dirs["pipeline_summary"] / "02_transcript_detection.json", transcript_records)
    if download_summary is not None:
        write_json(output_dirs["pipeline_summary"] / "03_download_summary.json", download_summary)
    if summary_stats is not None:
        write_json(output_dirs["pipeline_summary"] / "04_transcript_summarization.json", summary_stats)
    if topics_stats is not None:
        write_json(output_dirs["pipeline_summary"] / "05_topic_extraction.json", topics_stats)
    if digest_stats is not None:
        write_json(output_dirs["pipeline_summary"] / "06_perspective_digest_generation.json", digest_stats)
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
        "topics_result": topics_result,
        "topics_stats": topics_stats,
        "digest_result": digest_result,
        "digest_stats": digest_stats,
        "download_stats": download_stats,
        "output_root": str(output_dirs["run_root"]),
    }

    return pipeline_payload
