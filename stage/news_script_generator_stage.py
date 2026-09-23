"""Stage runner for generating perspective digests from topics."""

from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.shared_helper import setup_logging
from helpers.news_script_generator_helper import (
    build_digest_stats,
    generate_perspective_digest,
    load_topics,
    write_perspective_digest,
)


def run_stage(
    topics_file: str | Path,
    output_dir: str | Path,
    api_key: str | None = None,
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
        topics_file (str | Path): Path to topics.json.
        output_dir (str | Path): Directory to save digest files.
        api_key (str | None): DeepSeek API key.
        model (str): Primary model name.
        fallback_model (str): Backup model name.
        temperature (float): Sampling temperature (higher for creativity).
        max_tokens (int): Maximum output tokens.
        max_retries (int): Maximum retry count.
        backoff_seconds (int): Base retry delay.

    Returns:
        dict[str, Any]: Stage results with digest and stats.

    Example:
        >>> result = run_stage("topics/topics.json", "news_script")
        >>> "digest" in result
        True
    """
    # Load topics
    try:
        topics_data = load_topics(topics_file)
    except FileNotFoundError as exc:
        stats = build_digest_stats(0, 0, 0, 0, 0.0)
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
        output_dir,
        topics_data,
        tokens_used,
        cost,
    )

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

    return {
        "digest": digest_text,
        "word_count": word_count,
        "tokens_used": tokens_used,
        "cost": cost,
        "text_output_path": str(text_path),
        "json_output_path": str(json_path),
        "stats": stats,
    }


def main() -> None:
    """
    CLI entry point for the perspective digest generation stage.

    Arguments:
        None

    Returns:
        None

    Example:
        $ python stage/news_script_generator_stage.py --topics-file output_agenttube/2026-08-04/topics/topics.json --output-dir output_agenttube/2026-08-04/news_script
    """
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description="Run the perspective digest generation stage")
    parser.add_argument(
        "--topics-file",
        required=True,
        help="Path to topics.json",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save script files",
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
        default=0.7,
        help="Sampling temperature (higher = more creative)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8000,
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
        topics_file=args.topics_file,
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

    print(f"Word count: {result['word_count']}")
    print(f"Cost: ${result['cost']:.6f}")
    print(f"Text output: {result['text_output_path']}")
    print(f"JSON output: {result['json_output_path']}")


if __name__ == "__main__":
    main()