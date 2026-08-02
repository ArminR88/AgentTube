"""Stage runner for transcript fact checking."""

import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.shared_helper import setup_logging  # noqa: E402
from helpers.transcript_fact_checking_helper import (  # noqa: E402
    build_fact_check_stats,
    fact_check_transcript_records,
    write_fact_check_records,
)


DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
SEARCH_API_KEY_ENV = "SEARCH_API_KEY"


def get_api_key() -> str | None:
    """
    Read the DeepSeek API key from the environment.

    Arguments:
        None

    Returns:
        str | None: API key value or None if not set.
    """
    api_key = os.environ.get(DEEPSEEK_API_KEY_ENV)

    return api_key


def get_search_api_key() -> str | None:
    """
    Read the optional search API key from the environment.

    Arguments:
        None

    Returns:
        str | None: Search API key value or None if not set.
    """
    api_key = os.environ.get(SEARCH_API_KEY_ENV)

    return api_key


def run_stage(
    records: list[dict[str, Any]],
    output_dir: str = "fact_checking",
    search_api_key: str | None = None,
    search_provider: str | None = None,
) -> list[dict[str, Any]]:
    """
    Run the fact-check stage for summary transcript records.

    Arguments:
        records (list[dict[str, Any]]): Summary transcript records.
        output_dir (str): Directory to save fact-check files.
        search_api_key (str | None): Optional DeepSeek API key that enables Responses API web search.
        search_provider (str | None): Reserved compatibility argument for caller-provided provider names.

    Returns:
        list[dict[str, Any]]: Fact-check records.

    Example:
        >>> run_stage([])
        []
    """
    api_key = get_api_key() or search_api_key or get_search_api_key()
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY or SEARCH_API_KEY is not set.")

    fact_check_records = fact_check_transcript_records(
        records,
        api_key,
        search_api_key=search_api_key or get_search_api_key(),
        search_provider=search_provider,
    )
    write_fact_check_records(fact_check_records, output_dir)

    return fact_check_records


def main() -> None:
    """
    Run the CLI entry point.

    Arguments:
        None

    Returns:
        None

    Example:
        $ python stage/transcript_fact_checking_stage.py summary.json --api-key key
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run the transcript fact-checking stage")
    parser.add_argument("records_file", help="Path to a JSON file with summary transcript records")
    parser.add_argument("--output-dir", default="fact_checking", help="Directory for fact-check files")
    parser.add_argument("--search-api-key", help="Optional DeepSeek API key for web-assisted fact checking")
    parser.add_argument(
        "--search-provider",
        default="deepseek",
        choices=("deepseek", "tavily", "google", "bing"),
        help="Reserved compatibility argument for future search workflows",
    )
    parser.add_argument("--json", action="store_true", help="Print the fact-check records as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    with open(args.records_file, "r", encoding="utf-8") as file:
        records = json.load(file)

    fact_check_records = run_stage(
        records,
        output_dir=args.output_dir,
        search_api_key=args.search_api_key,
        search_provider=args.search_provider,
    )

    if args.json:
        print(json.dumps(fact_check_records, indent=2, default=str))
        return

    for index, record in enumerate(fact_check_records, 1):
        print(f"{index}. {record['channel_name']} | {record['title']} | bullets: {len(record.get('fact_checks') or [])}")


if __name__ == "__main__":
    main()