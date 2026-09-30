"""Stage runner for extracting topics across videos."""

from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.output_helper import build_output_directories, build_topics_output_directories, write_json
from helpers.summary_of_summaries_helper import (
    extract_topics,
    load_summary_files,
    write_topics,
    build_topics_stats,
)


def run_stage(
    model: str = "deepseek-v4-flash",
    fallback_model: str = "deepseek-chat",
    temperature: float = 0.3,
    max_tokens: int = 6000,
    max_retries: int = 3,
    backoff_seconds: int = 1,
) -> dict[str, Any]:
    """
    Run the topic extraction stage.

    Arguments:
        model (str): Primary model name.
        fallback_model (str): Backup model name.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum output tokens.
        max_retries (int): Maximum retry count.
        backoff_seconds (int): Base retry delay.

    Returns:
        dict[str, Any]: Stage results with topics and stats.

    Example:
        >>> result = run_stage()
        >>> "topics" in result
        True
    """
    # Load summary inputs produced by the previous stage.
    output_dirs = build_output_directories()
    claims_dirs = build_topics_output_directories()
    summary_records = load_summary_files(output_dirs["transcript_summary"])

    has_no_summary_records = not summary_records
    if has_no_summary_records == True:
        empty_stats = {
            "stage": "topic_extraction",
            "summary_record_count": 0,
            "topic_count": 0,
            "perspective_count": 0,
            "theme_count": 0,
            "error": "No summary files found",
            "generated_at": datetime.now().isoformat(),
        }
        write_json(output_dirs["pipeline_summary"] / "05_topic_extraction.json", empty_stats)
        empty_result = {
            "topics": [],
            "topic_count": 0,
            "perspective_count": 0,
            "tokens_used": 0,
            "cost": 0.0,
            "output_path": "",
            "stats": empty_stats,
            "summary_record_count": 0,
            "error": "No summary files found",
        }

        return empty_result

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set.")

    # Extract topics from the summary corpus.
    topics_draft, tokens_used, cost = extract_topics(
        summary_records,
        api_key=api_key,
        model=model,
        fallback_model=fallback_model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )

    # Persist extracted topics for downstream stages.
    topics_path = write_topics(topics_draft, claims_dirs["topics"], tokens_used, cost)

    summary_record_count = len(summary_records)
    stats = build_topics_stats(summary_record_count, topics_draft)
    write_json(output_dirs["pipeline_summary"] / "05_topic_extraction.json", stats)

    # Build API-safe topic payloads and accumulate perspective totals in one pass.
    topics_payload = []
    perspective_count = 0
    for topic in topics_draft.topics:
        has_model_dump = hasattr(topic, "model_dump")
        if has_model_dump == True:
            topic_payload = topic.model_dump()
        else:
            topic_payload = topic.dict()

        topics_payload.append(topic_payload)

        perspective_total_for_topic = len(topic.perspectives)
        perspective_count += perspective_total_for_topic

    result = {
        "topics": topics_payload,
        "topic_count": len(topics_draft.topics),
        "perspective_count": perspective_count,
        "tokens_used": tokens_used,
        "cost": cost,
        "output_path": str(topics_path),
        "stats": stats,
        "summary_record_count": summary_record_count,
    }

    return result