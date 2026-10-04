"""Stage runner for generating perspective digests from topics."""

from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.output_helper import (
    build_output_directories,
    build_topics_output_directories,
    write_json,
)
from helpers.shared_helper import load_json_object
from helpers.news_script_generator_helper import (
    build_digest_stats,
    generate_perspective_digest,
    write_perspective_digest,
)


def run_stage(
    model: str = "deepseek-v4-flash",
    fallback_model: str = "deepseek-chat",
    temperature: float = 0.7,
    max_tokens: int = 8000,
    max_retries: int = 3,
    backoff_seconds: int = 1,
) -> dict[str, Any]:
    """
    Run the perspective digest generation stage.

    Arguments:
        model (str): Primary model name.
        fallback_model (str): Backup model name.
        temperature (float): Sampling temperature (higher for creativity).
        max_tokens (int): Maximum output tokens.
        max_retries (int): Maximum retry count.
        backoff_seconds (int): Base retry delay.

    Returns:
        dict[str, Any]: Stage results with digest and stats.

    Example:
        >>> result = run_stage()
        >>> "digest" in result
        True
    """
    output_dirs = build_output_directories()
    claims_dirs = build_topics_output_directories()
    topics_file = claims_dirs["topics"] / "topics.json"
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set.")

    # Load topics
    try:
        topics_data = load_json_object(topics_file, missing_label="Topics file")
    except FileNotFoundError as exc:
        stats = build_digest_stats(0, 0, 0, 0, 0.0)
        write_json(output_dirs["pipeline_summary"] / "06_perspective_digest_generation.json", stats)
        return {
            "digest": "",
            "word_count": 0,
            "tokens_used": 0,
            "cost": 0.0,
            "text_output_path": "",
            "json_output_path": "",
            "stats": stats,
            "error": str(exc),
        }

    if not topics_data.get("topics"):
        stats = build_digest_stats(0, 0, 0, 0, 0.0)
        write_json(output_dirs["pipeline_summary"] / "06_perspective_digest_generation.json", stats)
        return {
            "digest": "No topics available to generate digest.",
            "word_count": 0,
            "tokens_used": 0,
            "cost": 0.0,
            "text_output_path": "",
            "json_output_path": "",
            "stats": stats,
            "error": "No topics found in input file",
        }

    # Generate digest
    digest_text, tokens_used, cost = generate_perspective_digest(
        topics_data,
        api_key=api_key,
        model=model,
        fallback_model=fallback_model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )

    # Write digest files
    text_path, json_path = write_perspective_digest(
        digest_text,
        claims_dirs["news_script"],
        topics_data,
        tokens_used,
        cost,
    )

    archival_dir = Path("output_agenttube") / "news_summary_archival"
    run_date = datetime.now().strftime("%Y%m%d")
    run_time = datetime.now().strftime("%H%M%S")
    archival_filename = f"perspective_digest_{run_date}_{run_time}.txt"
    archival_path = archival_dir / archival_filename
    archival_path.write_text(digest_text, encoding="utf-8")

    # Build stats
    topics = topics_data.get("topics", [])
    topic_count = len(topics)
    perspective_count = sum(
        len(topic.get("perspectives", []))
        for topic in topics
        if isinstance(topic, dict)
    )
    word_count = len(digest_text.split())
    stats = build_digest_stats(topic_count, perspective_count, word_count, tokens_used, cost)
    write_json(output_dirs["pipeline_summary"] / "06_perspective_digest_generation.json", stats)

    return {
        "digest": digest_text,
        "word_count": word_count,
        "tokens_used": tokens_used,
        "cost": cost,
        "text_output_path": str(text_path),
        "json_output_path": str(json_path),
        "stats": stats,
    }