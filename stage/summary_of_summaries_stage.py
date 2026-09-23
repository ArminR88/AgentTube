"""Stage runner for extracting topics across videos."""

from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.shared_helper import setup_logging
from helpers.summary_of_summaries_helper import (
    extract_topics,
    load_summary_files,
    write_topics,
    build_topics_stats,
)


def run_stage(
    summary_dir: str | Path,
    output_dir: str | Path,
    api_key: str | None = None,
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
        summary_dir (str | Path): Directory containing _summary.json files.
        output_dir (str | Path): Directory to save topics.json.
        api_key (str | None): DeepSeek API key.
        model (str): Primary model name.
        fallback_model (str): Backup model name.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum output tokens.
        max_retries (int): Maximum retry count.
        backoff_seconds (int): Base retry delay.

    Returns:
        dict[str, Any]: Stage results with topics and stats.

    Example:
        >>> result = run_stage("transcript_summary", "claims")
        >>> "topics" in result
        True
    """
    # Load all summary files
    summary_records = load_summary_files(summary_dir)
    
    if not summary_records:
        empty_stats = {
            "stage": "topic_extraction",
            "summary_record_count": 0,
            "topic_count": 0,
            "perspective_count": 0,
            "theme_count": 0,
        }
        return {
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

    # Extract topics
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

    # Write topics
    topics_path = write_topics(topics_draft, output_dir, tokens_used, cost)

    # Build stats
    stats = build_topics_stats(len(summary_records), topics_draft)

    topics_payload = [
        topic.model_dump() if hasattr(topic, "model_dump") else topic.dict()
        for topic in topics_draft.topics
    ]
    perspective_count = sum(len(topic.perspectives) for topic in topics_draft.topics)

    return {
        "topics": topics_payload,
        "topic_count": len(topics_draft.topics),
        "perspective_count": perspective_count,
        "tokens_used": tokens_used,
        "cost": cost,
        "output_path": str(topics_path),
        "stats": stats,
        "summary_record_count": len(summary_records),
    }


def main() -> None:
    """
    CLI entry point for the summary-of-summaries stage.

    Arguments:
        None

    Returns:
        None

    Example:
        $ python stage/summary_of_summaries_stage.py --summary-dir output_agenttube/2026-08-04/transcript_summary --output-dir output_agenttube/2026-08-04/topics
    """
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description="Run the topic extraction stage")
    parser.add_argument(
        "--summary-dir",
        required=True,
        help="Directory containing _summary.json files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save topics.json",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DEEPSEEK_API_KEY"),
        help="DeepSeek API key (defaults to DEEPSEEK_API_KEY from environment)",
    )
    parser.add_argument(
        "--model",
        default="deepseek-v4-flash",
        help="Primary DeepSeek model name",
    )
    parser.add_argument(
        "--fallback-model",
        default="deepseek-chat",
        help="Backup DeepSeek model name",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=6000,
        help="Maximum output tokens",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    if not args.api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set.")

    result = run_stage(
        summary_dir=args.summary_dir,
        output_dir=args.output_dir,
        api_key=args.api_key,
        model=args.model,
        fallback_model=args.fallback_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    print(f"Summary records processed: {result['summary_record_count']}")
    print(f"Topics extracted: {result['topic_count']}")
    print(f"Perspectives: {result['perspective_count']}")
    print(f"Cost: ${result['cost']:.6f}")
    print(f"Output: {result['output_path']}")


if __name__ == "__main__":
    main()