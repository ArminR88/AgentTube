"""Thin AgentTube pipeline entrypoint."""

import os
import sys
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

load_dotenv()

from helpers.output_helper import get_run_date
from stage.pipeline_stage import run_stage as run_pipeline_stage
from stage.transcript_fact_checking_stage import run_stage as run_fact_check_stage


def _resolve_fact_check_input_path(input_value: str) -> Path:
    """
    Resolve a fact-check input path.

    Arguments:
        input_value (str): User-provided input path.

    Returns:
        Path: Resolved file path.

    Example:
        >>> _resolve_fact_check_input_path("summary.json")
        PosixPath('summary.json')
    """
    candidate_path = Path(input_value)
    if candidate_path.is_file():
        return candidate_path

    if not candidate_path.is_absolute():
        cwd_candidate = Path.cwd() / candidate_path
        if cwd_candidate.is_file():
            return cwd_candidate

        summary_candidate = Path.cwd() / "output_agenttube" / get_run_date() / "transcript_summary" / candidate_path.name
        if summary_candidate.is_file():
            return summary_candidate

    return candidate_path


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run the AgentTube pipeline")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--json", action="store_true", help="Print the detection schema as JSON")
    parser.add_argument("--no-detection", action="store_true", help="Skip the detection stage")
    parser.add_argument("--no-transcript-detection", action="store_true", help="Skip the transcript detection stage")
    parser.add_argument("--no-download", action="store_true", help="Skip transcript download after transcript detection")
    parser.add_argument("--no-summary", action="store_true", help="Skip transcript summarization after transcript download")
    parser.add_argument("--no-fact-check", action="store_true", help="Skip transcript fact checking after summarization")
    parser.add_argument("--fact-check-only", action="store_true", help="Run only fact checking on an existing summary JSON file")
    parser.add_argument("--fact-check-input", help="Path to a JSON file containing summary transcript records")
    parser.add_argument(
        "--search-api-key",
        default=os.environ.get("SEARCH_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"),
        help="Optional DeepSeek API key for web-assisted fact checking (defaults to SEARCH_API_KEY or DEEPSEEK_API_KEY from .env)",
    )
    parser.add_argument(
        "--search-provider",
        default="deepseek",
        choices=("deepseek", "tavily", "google", "bing"),
        help="Reserved compatibility argument for future search workflows",
    )
    parser.add_argument("--summary-limit", type=int, help="Limit the number of summarized records")
    args = parser.parse_args()

    if args.fact_check_only:
        if not args.fact_check_input:
            raise SystemExit("--fact-check-input is required when --fact-check-only is set.")

        import json as _json

        fact_check_input_path = _resolve_fact_check_input_path(args.fact_check_input)
        if not fact_check_input_path.exists():
            raise SystemExit(
                "Fact-check input file not found: "
                f"{args.fact_check_input}. "
                "Try output_agenttube/<today>/transcript_summary/<filename>.json"
            )

        with open(fact_check_input_path, "r", encoding="utf-8") as file:
            summary_records = _json.load(file)

        fact_check_records = run_fact_check_stage(
            summary_records,
            search_api_key=args.search_api_key,
            search_provider=args.search_provider,
        )

        payload = {
            "summary_records": summary_records,
            "fact_check_records": fact_check_records,
            "fact_check_only": True,
        }

        if args.json:
            print(_json.dumps(payload, indent=2, default=str))
            return

        print(f"Fact checks: {len(fact_check_records)} record(s) processed")
        return

    try:
        payload = run_pipeline_stage(
            verbose=args.verbose,
            run_detection=not args.no_detection,
            run_transcript_detection=not args.no_transcript_detection,
            run_download=not args.no_download,
            run_summary=not args.no_summary,
            run_fact_check=not args.no_fact_check,
            search_api_key=args.search_api_key,
            search_provider=args.search_provider,
            summary_limit=args.summary_limit,
        )
    except SystemExit as exc:
        sys.exit(2)

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return

    transcript_records = cast(list[dict[str, Any]], payload.get("transcript_records") or [])
    for index, record in enumerate(transcript_records, 1):
        transcript_state = "yes" if record.get("transcript_available") else "no"
        print(
            f"{index}. {record['channel_name']} | {record['title']} | duration: {record['duration']} | transcript: {transcript_state}"
        )

    summary_transcript_records = cast(list[dict[str, Any]], payload.get("summary_transcript_records") or [])
    if summary_transcript_records:
        print(f"Summaries: {len(summary_transcript_records)} record(s) processed")

    fact_check_records = cast(list[dict[str, Any]], payload.get("fact_check_records") or [])
    if fact_check_records:
        print(f"Fact checks: {len(fact_check_records)} record(s) processed")


if __name__ == "__main__":
    main()